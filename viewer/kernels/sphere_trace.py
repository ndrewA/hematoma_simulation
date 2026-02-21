"""Multi-surface SDF sphere tracing (raymarching) kernel."""

import taichi as ti
import taichi.math as tm


@ti.func
def trilinear_sample(vol: ti.template(), x: float, y: float, z: float) -> float:
    Ni, Nj, Nk = vol.shape[0], vol.shape[1], vol.shape[2]
    x = ti.max(0.0, ti.min(float(Ni - 1), x))
    y = ti.max(0.0, ti.min(float(Nj - 1), y))
    z = ti.max(0.0, ti.min(float(Nk - 1), z))

    x0, y0, z0 = int(ti.floor(x)), int(ti.floor(y)), int(ti.floor(z))
    x1 = ti.min(x0 + 1, Ni - 1)
    y1 = ti.min(y0 + 1, Nj - 1)
    z1 = ti.min(z0 + 1, Nk - 1)
    fx, fy, fz = x - float(x0), y - float(y0), z - float(z0)

    return (vol[x0, y0, z0] * (1-fx)*(1-fy)*(1-fz) +
            vol[x1, y0, z0] * fx*(1-fy)*(1-fz) +
            vol[x0, y1, z0] * (1-fx)*fy*(1-fz) +
            vol[x1, y1, z0] * fx*fy*(1-fz) +
            vol[x0, y0, z1] * (1-fx)*(1-fy)*fz +
            vol[x1, y0, z1] * fx*(1-fy)*fz +
            vol[x0, y1, z1] * (1-fx)*fy*fz +
            vol[x1, y1, z1] * fx*fy*fz)


@ti.func
def sdf_normal(vol: ti.template(), p: tm.vec3) -> tm.vec3:
    eps = 0.5
    nx = trilinear_sample(vol, p.x+eps, p.y, p.z) - trilinear_sample(vol, p.x-eps, p.y, p.z)
    ny = trilinear_sample(vol, p.x, p.y+eps, p.z) - trilinear_sample(vol, p.x, p.y-eps, p.z)
    nz = trilinear_sample(vol, p.x, p.y, p.z+eps) - trilinear_sample(vol, p.x, p.y, p.z-eps)
    n = tm.vec3(nx, ny, nz)
    length = tm.length(n)
    if length > 1e-8:
        n = n / length
    return n


@ti.func
def ray_aabb(eye: tm.vec3, rd: tm.vec3, N_f: float, max_dist: float):
    t_near = 0.0
    t_far = max_dist
    for axis in ti.static(range(3)):
        if ti.abs(rd[axis]) > 1e-8:
            t1 = (0.0 - eye[axis]) / rd[axis]
            t2 = (N_f - eye[axis]) / rd[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = ti.max(t_near, t1)
            t_far = ti.min(t_far, t2)
    return t_near, t_far


@ti.func
def trace_sdf(sdf: ti.template(), eye: tm.vec3, rd: tm.vec3,
              t_near: float, t_far: float, inv_dx: float, min_step: float):
    """Trace a single SDF field. Returns (hit, t_hit, hit_position)."""
    t = t_near + min_step
    hit = False
    hit_t = t_far
    hit_p = tm.vec3(0.0)
    prev_d = 1e6

    for _step in range(512):
        if t > t_far:
            break
        p = eye + rd * t
        d_mm = trilinear_sample(sdf, p.x, p.y, p.z)

        if d_mm < 0.2 and prev_d > 0.0:
            # Binary search for precise zero-crossing
            t_lo = t - ti.max(prev_d * inv_dx * 0.9, min_step)
            t_hi = t
            for _refine in range(8):
                t_mid = (t_lo + t_hi) * 0.5
                p_mid = eye + rd * t_mid
                d_mid = trilinear_sample(sdf, p_mid.x, p_mid.y, p_mid.z)
                if d_mid < 0.0:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            hit_t = (t_lo + t_hi) * 0.5
            hit_p = eye + rd * hit_t
            hit = True
            break

        step = ti.max(ti.abs(d_mm) * inv_dx * 0.9, min_step)
        prev_d = d_mm
        t += step

    return hit, hit_t, hit_p


@ti.func
def get_surface_color(p: tm.vec3, base: tm.vec3,
                      material_map: ti.template(), cat_lut: ti.template(),
                      t1w_field: ti.template(),
                      mat_on: int, t1w_on: int,
                      t1w_vmin: float, t1w_vmax: float) -> tm.vec3:
    """Determine surface color at hit point: base → material override → T1w override."""
    color = base
    if mat_on:
        ix = int(ti.round(p.x))
        iy = int(ti.round(p.y))
        iz = int(ti.round(p.z))
        ix = ti.max(0, ti.min(ix, material_map.shape[0] - 1))
        iy = ti.max(0, ti.min(iy, material_map.shape[1] - 1))
        iz = ti.max(0, ti.min(iz, material_map.shape[2] - 1))
        label = material_map[ix, iy, iz]
        rgba = cat_lut[int(label)]
        if rgba.w > 0.01:
            color = tm.vec3(rgba.x, rgba.y, rgba.z)
    if t1w_on:
        val = trilinear_sample(t1w_field, p.x, p.y, p.z)
        t = tm.clamp((val - t1w_vmin) / ti.max(t1w_vmax - t1w_vmin, 1e-6), 0.0, 1.0)
        color = tm.vec3(t, t, t)
    return color


@ti.func
def shade_surface(color: tm.vec3, n: tm.vec3, rd: tm.vec3) -> tm.vec3:
    """Headlight + fill light + specular shading."""
    ambient = 0.20
    spec_power = 32.0
    head_dir = -rd
    diffuse = ti.max(0.0, tm.dot(n, head_dir))
    fill_dir = tm.normalize(tm.vec3(0.3, -0.2, 1.0))
    fill = ti.max(0.0, tm.dot(n, fill_dir)) * 0.3
    half_vec = head_dir  # normalize(head_dir + head_dir) = head_dir for headlight
    spec = ti.pow(ti.max(0.0, tm.dot(n, half_vec)), spec_power) * 0.3
    lighting = ambient + diffuse * 0.6 + fill + spec
    return tm.clamp(color * lighting, 0.0, 1.0)


@ti.kernel
def sphere_trace(
    skull_sdf: ti.template(),
    brain_sdf: ti.template(),
    material_map: ti.template(),
    cat_lut: ti.template(),
    t1w_field: ti.template(),
    buf: ti.template(),
    px0: int, py0: int, pw: int, ph: int,
    eye_x: float, eye_y: float, eye_z: float,
    fwd_x: float, fwd_y: float, fwd_z: float,
    right_x: float, right_y: float, right_z: float,
    up_x: float, up_y: float, up_z: float,
    fov_scale: float,
    vol_N: int,
    dx_mm: float,
    skull_on: int, brain_on: int,
    mat_on: int, t1w_on: int,
    skull_opacity: float, brain_opacity: float,
    t1w_vmin: float, t1w_vmax: float,
):
    eye = tm.vec3(eye_x, eye_y, eye_z)
    fwd = tm.vec3(fwd_x, fwd_y, fwd_z)
    right = tm.vec3(right_x, right_y, right_z)
    up = tm.vec3(up_x, up_y, up_z)

    bg_color = tm.vec3(0.12, 0.12, 0.16)
    skull_base = tm.vec3(0.92, 0.87, 0.78)   # bone beige
    brain_base = tm.vec3(0.75, 0.72, 0.70)   # warm gray
    inv_dx = 1.0 / dx_mm
    min_step = 0.3
    max_dist = float(vol_N) * 3.0
    N_f = float(vol_N)

    for px, py in ti.ndrange(pw, ph):
        u_ndc = (2.0 * (px + 0.5) / float(pw) - 1.0) * fov_scale * float(pw) / float(ph)
        v_ndc = (2.0 * (py + 0.5) / float(ph) - 1.0) * fov_scale
        rd = tm.normalize(fwd + right * u_ndc + up * v_ndc)

        t_near, t_far = ray_aabb(eye, rd, N_f, max_dist)

        color = bg_color
        if t_near < t_far:
            # Trace each active surface independently
            s_hit = False
            s_t = max_dist
            s_p = tm.vec3(0.0)
            b_hit = False
            b_t = max_dist
            b_p = tm.vec3(0.0)

            if skull_on:
                s_hit, s_t, s_p = trace_sdf(
                    skull_sdf, eye, rd, t_near, t_far, inv_dx, min_step)
            if brain_on:
                b_hit, b_t, b_p = trace_sdf(
                    brain_sdf, eye, rd, t_near, t_far, inv_dx, min_step)

            # Shade each hit
            s_color = bg_color
            b_color = bg_color
            if s_hit:
                n = sdf_normal(skull_sdf, s_p)
                if tm.dot(n, rd) > 0.0:
                    n = -n
                # Skull: no material map (would sample brain tissue through thin bone)
                sc = get_surface_color(
                    s_p, skull_base, material_map, cat_lut, t1w_field,
                    0, t1w_on, t1w_vmin, t1w_vmax)
                s_color = shade_surface(sc, n, rd)
            if b_hit:
                n = sdf_normal(brain_sdf, b_p)
                if tm.dot(n, rd) > 0.0:
                    n = -n
                # Brain: material map applies here
                bc = get_surface_color(
                    b_p, brain_base, material_map, cat_lut, t1w_field,
                    mat_on, t1w_on, t1w_vmin, t1w_vmax)
                b_color = shade_surface(bc, n, rd)

            # Front-to-back compositing
            if s_hit and b_hit:
                front_c = s_color
                front_a = skull_opacity
                back_c = b_color
                back_a = brain_opacity
                if s_t > b_t:
                    front_c = b_color
                    front_a = brain_opacity
                    back_c = s_color
                    back_a = skull_opacity
                color = (front_c * front_a
                         + back_c * back_a * (1.0 - front_a)
                         + bg_color * (1.0 - front_a) * (1.0 - back_a))
            elif s_hit:
                color = s_color * skull_opacity + bg_color * (1.0 - skull_opacity)
            elif b_hit:
                color = b_color * brain_opacity + bg_color * (1.0 - brain_opacity)

        buf[px0 + px, py0 + py] = color

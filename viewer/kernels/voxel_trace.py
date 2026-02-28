"""Voxel DDA ray tracer with semi-transparent material groups."""

import taichi as ti
import taichi.math as tm

# Material group indices
GROUP_EMPTY = -1
GROUP_BRAIN = 0      # labels 1-6 (WM, GM, deep GM, cerebellar, brainstem)
GROUP_CSF = 1        # labels 7, 8 (ventricular, subarachnoid)
GROUP_DURA = 2       # label 10 (dural membrane)
GROUP_SKULL = 3      # virtual label 12 (material==0, skull_sdf<0)
GROUP_CHOROID = 4    # label 9 (choroid plexus)
GROUP_VESSEL = 5     # label 11 (vessel / venous sinus)
N_GROUPS = 6


@ti.func
def label_to_group(label: int) -> int:
    group = GROUP_EMPTY
    if 1 <= label <= 6:
        group = GROUP_BRAIN
    elif label == 7 or label == 8:
        group = GROUP_CSF
    elif label == 10:
        group = GROUP_DURA
    elif label == 9:
        group = GROUP_CHOROID
    elif label == 11:
        group = GROUP_VESSEL
    elif label == 12:
        group = GROUP_SKULL
    return group


@ti.func
def ray_aabb(eye: tm.vec3, rd: tm.vec3, N_f: float, max_dist: float):
    """Ray-AABB intersection for [0, N]^3. Returns (t_near, t_far, entry_axis, entry_sign)."""
    t_near = 0.0
    t_far = max_dist
    entry_axis = 0
    entry_sign = 1.0
    for axis in ti.static(range(3)):
        if ti.abs(rd[axis]) > 1e-8:
            t1 = (0.0 - eye[axis]) / rd[axis]
            t2 = (N_f - eye[axis]) / rd[axis]
            sign_val = -1.0  # entering low face
            if t1 > t2:
                t1, t2 = t2, t1
                sign_val = 1.0  # entering high face
            if t1 > t_near:
                t_near = t1
                entry_axis = axis
                entry_sign = sign_val
            t_far = ti.min(t_far, t2)
    return t_near, t_far, entry_axis, entry_sign


@ti.func
def shade_voxel(color: tm.vec3, n: tm.vec3, rd: tm.vec3) -> tm.vec3:
    """Flat-face shading: ambient + headlight diffuse + fill light."""
    ambient = 0.25
    head_dir = -rd
    diffuse = ti.max(0.0, tm.dot(n, head_dir)) * 0.55
    fill_dir = tm.normalize(tm.vec3(0.3, -0.2, 1.0))
    fill = ti.max(0.0, tm.dot(n, fill_dir)) * 0.25
    lighting = ambient + diffuse + fill
    return tm.clamp(color * lighting, 0.0, 1.0)


@ti.kernel
def voxel_trace(
    material_map: ti.template(),
    skull_sdf: ti.template(),
    cat_lut: ti.template(),
    group_opacity: ti.template(),
    buf: ti.template(),
    px0: int, py0: int, pw: int, ph: int,
    eye_x: float, eye_y: float, eye_z: float,
    fwd_x: float, fwd_y: float, fwd_z: float,
    right_x: float, right_y: float, right_z: float,
    up_x: float, up_y: float, up_z: float,
    fov_scale: float,
    vol_N: int,
):
    eye = tm.vec3(eye_x, eye_y, eye_z)
    fwd = tm.vec3(fwd_x, fwd_y, fwd_z)
    right = tm.vec3(right_x, right_y, right_z)
    up = tm.vec3(up_x, up_y, up_z)

    bg = tm.vec3(0.12, 0.12, 0.16)
    N_f = float(vol_N)
    N_i = vol_N
    max_dist = N_f * 3.0

    for px, py in ti.ndrange(pw, ph):
        u_ndc = (2.0 * (px + 0.5) / float(pw) - 1.0) * fov_scale * float(pw) / float(ph)
        v_ndc = (2.0 * (py + 0.5) / float(ph) - 1.0) * fov_scale
        rd = tm.normalize(fwd + right * u_ndc + up * v_ndc)

        t_near, t_far, entry_axis, entry_sign = ray_aabb(eye, rd, N_f, max_dist)

        color = bg
        if t_near < t_far:
            # Entry point (small offset to land inside AABB)
            p = eye + rd * (t_near + 0.001)

            # Starting voxel
            ix = ti.max(0, ti.min(int(ti.floor(p.x)), N_i - 1))
            iy = ti.max(0, ti.min(int(ti.floor(p.y)), N_i - 1))
            iz = ti.max(0, ti.min(int(ti.floor(p.z)), N_i - 1))

            # DDA step directions
            step_x = 1 if rd.x >= 0.0 else -1
            step_y = 1 if rd.y >= 0.0 else -1
            step_z = 1 if rd.z >= 0.0 else -1

            # Parameter distance per voxel step on each axis
            t_delta_x = ti.abs(1.0 / rd.x) if ti.abs(rd.x) > 1e-8 else 1e30
            t_delta_y = ti.abs(1.0 / rd.y) if ti.abs(rd.y) > 1e-8 else 1e30
            t_delta_z = ti.abs(1.0 / rd.z) if ti.abs(rd.z) > 1e-8 else 1e30

            # Parameter distance to next boundary on each axis
            t_max_x = 1e30
            t_max_y = 1e30
            t_max_z = 1e30
            if ti.abs(rd.x) > 1e-8:
                next_x = float(ix + 1) if step_x > 0 else float(ix)
                t_max_x = (next_x - p.x) / rd.x
            if ti.abs(rd.y) > 1e-8:
                next_y = float(iy + 1) if step_y > 0 else float(iy)
                t_max_y = (next_y - p.y) / rd.y
            if ti.abs(rd.z) > 1e-8:
                next_z = float(iz + 1) if step_z > 0 else float(iz)
                t_max_z = (next_z - p.z) / rd.z

            # Initial face normal from AABB entry
            face_n = tm.vec3(0.0, 0.0, 0.0)
            if entry_axis == 0:
                face_n = tm.vec3(entry_sign, 0.0, 0.0)
            elif entry_axis == 1:
                face_n = tm.vec3(0.0, entry_sign, 0.0)
            else:
                face_n = tm.vec3(0.0, 0.0, entry_sign)

            # Front-to-back compositing state
            acc_color = tm.vec3(0.0)
            acc_alpha = 0.0
            prev_group = GROUP_EMPTY

            for _step in range(N_i * 3):
                if ix < 0 or ix >= N_i or iy < 0 or iy >= N_i or iz < 0 or iz >= N_i:
                    break
                if acc_alpha > 0.99:
                    break

                # Determine material label and group
                mat = int(material_map[ix, iy, iz])
                label = mat
                group = label_to_group(label)
                if group == GROUP_EMPTY:
                    # Skull detection: zero-crossing of skull_sdf at this voxel
                    # Check 6 face-neighbors for sign change (same as 2D contour)
                    s = skull_sdf[ix, iy, iz]
                    on_skull = False
                    if ix > 0 and s * skull_sdf[ix-1, iy, iz] < 0.0:
                        on_skull = True
                    if ix < N_i-1 and s * skull_sdf[ix+1, iy, iz] < 0.0:
                        on_skull = True
                    if iy > 0 and s * skull_sdf[ix, iy-1, iz] < 0.0:
                        on_skull = True
                    if iy < N_i-1 and s * skull_sdf[ix, iy+1, iz] < 0.0:
                        on_skull = True
                    if iz > 0 and s * skull_sdf[ix, iy, iz-1] < 0.0:
                        on_skull = True
                    if iz < N_i-1 and s * skull_sdf[ix, iy, iz+1] < 0.0:
                        on_skull = True
                    if on_skull:
                        label = 12
                        group = GROUP_SKULL

                # Composite non-empty voxels
                if group != GROUP_EMPTY:
                    a = group_opacity[group]
                    # Skip opaque interior (already fully composited at entry)
                    if a > 0.0 and not (group == prev_group and a >= 1.0):
                        rgba = cat_lut[label]
                        voxel_color = tm.vec3(rgba.x, rgba.y, rgba.z)
                        shaded = voxel_color * 0.35
                        if group != prev_group:
                            shaded = shade_voxel(voxel_color, face_n, rd)
                        acc_color += shaded * a * (1.0 - acc_alpha)
                        acc_alpha += a * (1.0 - acc_alpha)

                prev_group = group

                # DDA step: advance to next voxel boundary
                if t_max_x < t_max_y and t_max_x < t_max_z:
                    ix += step_x
                    t_max_x += t_delta_x
                    face_n = tm.vec3(-float(step_x), 0.0, 0.0)
                elif t_max_y < t_max_z:
                    iy += step_y
                    t_max_y += t_delta_y
                    face_n = tm.vec3(0.0, -float(step_y), 0.0)
                else:
                    iz += step_z
                    t_max_z += t_delta_z
                    face_n = tm.vec3(0.0, 0.0, -float(step_z))

            # Blend accumulated color with background
            color = acc_color + bg * (1.0 - acc_alpha)

        buf[px0 + px, py0 + py] = color

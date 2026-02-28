"""3D crosshair orientation widget: three colored planes + wireframe cube."""

import taichi as ti
import taichi.math as tm


# Plane colors (Sagittal=red, Coronal=green, Axial=blue)
_PLANE_R = tm.vec3(0.8, 0.2, 0.2)
_PLANE_G = tm.vec3(0.2, 0.8, 0.2)
_PLANE_B = tm.vec3(0.3, 0.4, 1.0)
_PLANE_ALPHA = 0.25
_LINE_ALPHA = 0.7
_LINE_W = 0.8        # voxel units — width of crosshair lines on planes
_EDGE_COLOR = tm.vec3(0.45, 0.45, 0.50)
_EDGE_ALPHA = 0.6
_EDGE_W = 1.0        # voxel units — width of cube wireframe edges


@ti.func
def _ray_aabb(eye: tm.vec3, rd: tm.vec3, N_f: float):
    """Ray-AABB intersection for [0, N]^3. Returns (t_near, t_far, near_axis, far_axis)."""
    t_near = -1e30
    t_far = 1e30
    near_axis = 0
    far_axis = 0
    for axis in ti.static(range(3)):
        if ti.abs(rd[axis]) > 1e-8:
            t1 = (0.0 - eye[axis]) / rd[axis]
            t2 = (N_f - eye[axis]) / rd[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_near:
                t_near = t1
                near_axis = axis
            if t2 < t_far:
                t_far = t2
                far_axis = axis
        else:
            if eye[axis] < 0.0 or eye[axis] > N_f:
                t_near = 1.0
                t_far = -1.0
    return t_near, t_far, near_axis, far_axis


@ti.func
def _in_box(p: tm.vec3, N_f: float) -> bool:
    return (p.x >= 0.0 and p.x <= N_f and
            p.y >= 0.0 and p.y <= N_f and
            p.z >= 0.0 and p.z <= N_f)


@ti.func
def _near_crosshair_line(p: tm.vec3, cross: tm.vec3, plane_axis: int, w: float) -> bool:
    """Check if point on a crosshair plane is near one of the two crosshair lines."""
    near = False
    for a in ti.static(range(3)):
        if a != plane_axis:
            if ti.abs(p[a] - cross[a]) < w:
                near = True
    return near


@ti.func
def _near_cube_edge(p: tm.vec3, face_axis: int, N_f: float, w: float) -> bool:
    """Check if point on an AABB face is near a wireframe edge."""
    near = False
    for a in ti.static(range(3)):
        if a != face_axis:
            if p[a] < w or p[a] > N_f - w:
                near = True
    return near


@ti.kernel
def crosshair_3d(
    buf: ti.template(),
    px0: int, py0: int, pw: int, ph: int,
    eye_x: float, eye_y: float, eye_z: float,
    fwd_x: float, fwd_y: float, fwd_z: float,
    right_x: float, right_y: float, right_z: float,
    up_x: float, up_y: float, up_z: float,
    fov_scale: float,
    vol_N: int,
    cross_x: float, cross_y: float, cross_z: float,
):
    eye = tm.vec3(eye_x, eye_y, eye_z)
    fwd = tm.vec3(fwd_x, fwd_y, fwd_z)
    right = tm.vec3(right_x, right_y, right_z)
    up = tm.vec3(up_x, up_y, up_z)
    cross = tm.vec3(cross_x, cross_y, cross_z)
    N_f = float(vol_N)

    plane_colors = tm.mat3(_PLANE_R, _PLANE_G, _PLANE_B)  # rows = planes

    bw = buf.shape[0]
    bh = buf.shape[1]

    for px, py in ti.ndrange(pw, ph):
        u_ndc = (2.0 * (px + 0.5) / float(pw) - 1.0) * fov_scale * float(pw) / float(ph)
        v_ndc = (2.0 * (py + 0.5) / float(ph) - 1.0) * fov_scale
        rd = tm.normalize(fwd + right * u_ndc + up * v_ndc)

        t_near, t_far, near_axis, far_axis = _ray_aabb(eye, rd, N_f)

        if t_near < t_far:
            # Collect hits: up to 5 (3 planes + 2 wireframe faces)
            # Each hit: [t, r, g, b, a]
            hits = ti.Matrix.zero(float, 5, 5)
            for i in ti.static(range(5)):
                hits[i, 0] = 1e30  # sentinel
            n = 0

            # --- 3 crosshair planes ---
            for axis in ti.static(range(3)):
                d = rd[axis]
                if ti.abs(d) > 1e-8:
                    t = (cross[axis] - eye[axis]) / d
                    if t > 0.0:
                        p = eye + rd * t
                        if _in_box(p, N_f):
                            a = _PLANE_ALPHA
                            if _near_crosshair_line(p, cross, axis, _LINE_W):
                                a = _LINE_ALPHA
                            c = tm.vec3(plane_colors[axis, 0],
                                        plane_colors[axis, 1],
                                        plane_colors[axis, 2])
                            hits[n, 0] = t
                            hits[n, 1] = c.x
                            hits[n, 2] = c.y
                            hits[n, 3] = c.z
                            hits[n, 4] = a
                            n += 1

            # --- Wireframe cube edges at AABB entry/exit ---
            if t_near > 0.0:
                p_entry = eye + rd * t_near
                if _near_cube_edge(p_entry, near_axis, N_f, _EDGE_W):
                    hits[n, 0] = t_near
                    hits[n, 1] = _EDGE_COLOR.x
                    hits[n, 2] = _EDGE_COLOR.y
                    hits[n, 3] = _EDGE_COLOR.z
                    hits[n, 4] = _EDGE_ALPHA
                    n += 1

            if t_far > 0.0 and n < 5:
                p_exit = eye + rd * t_far
                if _near_cube_edge(p_exit, far_axis, N_f, _EDGE_W):
                    hits[n, 0] = t_far
                    hits[n, 1] = _EDGE_COLOR.x
                    hits[n, 2] = _EDGE_COLOR.y
                    hits[n, 3] = _EDGE_COLOR.z
                    hits[n, 4] = _EDGE_ALPHA
                    n += 1

            # --- Sort by t (bubble sort, compile-time unrolled) ---
            for i in ti.static(range(4)):
                for j in ti.static(range(4 - i)):
                    if hits[j, 0] > hits[j + 1, 0]:
                        for c in ti.static(range(5)):
                            tmp = hits[j, c]
                            hits[j, c] = hits[j + 1, c]
                            hits[j + 1, c] = tmp

            # --- Front-to-back composite ---
            acc_color = tm.vec3(0.0)
            acc_alpha = 0.0

            for i in ti.static(range(5)):
                t_val = hits[i, 0]
                if t_val < 1e29:
                    ha = hits[i, 4]
                    weight = ha * (1.0 - acc_alpha)
                    acc_color += tm.vec3(hits[i, 1], hits[i, 2], hits[i, 3]) * weight
                    acc_alpha += weight

            # Blend onto existing buffer content
            bx = px0 + px
            by = py0 + py
            if 0 <= bx < bw and 0 <= by < bh:
                old = buf[bx, by]
                buf[bx, by] = acc_color + old * (1.0 - acc_alpha)

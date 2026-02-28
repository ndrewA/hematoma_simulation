"""DEC (direction-encoded color) slice kernel for fiber tensor data."""

import taichi as ti
import taichi.math as tm


@ti.func
def _power_iteration(M00: float, M11: float, M22: float,
                     M01: float, M02: float, M12: float) -> tm.vec3:
    """Principal eigenvector of 3x3 symmetric matrix via power iteration."""
    # Start with a non-degenerate initial vector
    v = tm.vec3(1.0, 0.5, 0.25)
    v = v / tm.length(v)

    for _ in range(5):
        # Matrix-vector multiply: M @ v
        nv = tm.vec3(
            M00 * v.x + M01 * v.y + M02 * v.z,
            M01 * v.x + M11 * v.y + M12 * v.z,
            M02 * v.x + M12 * v.y + M22 * v.z,
        )
        length = tm.length(nv)
        if length > 1e-12:
            v = nv / length
        else:
            break
    return v


@ti.kernel
def dec_slice(
    fiber: ti.template(),    # Vector.field(6, f32, (Fi, Fj, Fk))
    buf: ti.template(),      # vec3 composite buffer
    axis: int,
    slice_idx: int,          # slice index in GRID coords
    px0: int, py0: int,
    pw: int, ph: int,
    opacity: float,
    zoom: float,
    pan_x: float, pan_y: float,
    # Grid-to-fiber coordinate transform (affine 3x4)
    g2f_00: float, g2f_01: float, g2f_02: float, g2f_03: float,
    g2f_10: float, g2f_11: float, g2f_12: float, g2f_13: float,
    g2f_20: float, g2f_21: float, g2f_22: float, g2f_23: float,
    # Grid dimensions for pixel mapping
    grid_N: int,
):
    """Render DEC-colored fiber orientation slice."""
    Fi = fiber.shape[0]
    Fj = fiber.shape[1]
    Fk = fiber.shape[2]

    dim_u = grid_N
    dim_v = grid_N
    bw = buf.shape[0]
    bh = buf.shape[1]

    for px, py in ti.ndrange(pw, ph):
        scale = ti.min(float(pw) / float(dim_u), float(ph) / float(dim_v)) * zoom
        u = (px - pw / 2.0) / scale + dim_u / 2.0 - pan_x
        v = (py - ph / 2.0) / scale + dim_v / 2.0 - pan_y

        ui = int(ti.round(u))
        vi = int(ti.round(v))

        if 0 <= ui < dim_u and 0 <= vi < dim_v:
            # Reconstruct 3D grid coordinate
            gi = 0.0
            gj = 0.0
            gk = 0.0
            if axis == 2:  # axial
                gi = float(ui)
                gj = float(vi)
                gk = float(slice_idx)
            elif axis == 1:  # coronal
                gi = float(ui)
                gj = float(slice_idx)
                gk = float(vi)
            else:  # sagittal
                gi = float(slice_idx)
                gj = float(ui)
                gk = float(vi)

            # Transform grid coord → fiber voxel coord
            fi = g2f_00 * gi + g2f_01 * gj + g2f_02 * gk + g2f_03
            fj = g2f_10 * gi + g2f_11 * gj + g2f_12 * gk + g2f_13
            fk = g2f_20 * gi + g2f_21 * gj + g2f_22 * gk + g2f_23

            # Nearest-neighbor sampling in fiber space
            fii = int(ti.round(fi))
            fjj = int(ti.round(fj))
            fkk = int(ti.round(fk))

            if 0 <= fii < Fi and 0 <= fjj < Fj and 0 <= fkk < Fk:
                t = fiber[fii, fjj, fkk]
                M00, M11, M22 = t[0], t[1], t[2]
                M01, M02, M12 = t[3], t[4], t[5]

                trace = M00 + M11 + M22
                if trace > 1e-8:
                    eigvec = _power_iteration(M00, M11, M22, M01, M02, M12)
                    # DEC color: |eigvec components| scaled by brightness
                    brightness = ti.min(trace * 3.0, 1.0)
                    color = tm.vec3(
                        ti.abs(eigvec.x),
                        ti.abs(eigvec.y),
                        ti.abs(eigvec.z),
                    ) * brightness

                    a = opacity
                    bx = px0 + px
                    by = py0 + py
                    if 0 <= bx < bw and 0 <= by < bh:
                        old = buf[bx, by]
                        buf[bx, by] = old * (1.0 - a) + color * a

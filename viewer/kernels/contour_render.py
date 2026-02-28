"""Zero-contour rendering kernel for signed distance fields."""

import taichi as ti

from viewer.kernels.common import get_slice_dims as _get_slice_dims


@ti.func
def _sample_sdf(vol: ti.template(), axis: int, slice_idx: int, ui: int, vi: int) -> float:
    val = 0.0
    if axis == 2:
        val = vol[ui, vi, slice_idx]
    elif axis == 1:
        val = vol[ui, slice_idx, vi]
    else:
        val = vol[slice_idx, ui, vi]
    return val


@ti.kernel
def contour_slice(
    vol: ti.template(),      # f32 3D SDF field
    buf: ti.template(),      # vec3 composite buffer
    axis: int,
    slice_idx: int,
    px0: int, py0: int,
    pw: int, ph: int,
    opacity: float,
    zoom: float,
    pan_x: float, pan_y: float,
    # Contour color (RGB)
    cr: float, cg: float, cb: float,
):
    """Draw zero-contour of SDF as a colored line on the slice."""
    dim_u, dim_v = _get_slice_dims(vol, axis)
    contour_color = ti.Vector([cr, cg, cb])
    bw = buf.shape[0]
    bh = buf.shape[1]

    for px, py in ti.ndrange(pw, ph):
        scale = ti.min(float(pw) / float(dim_u), float(ph) / float(dim_v)) * zoom
        u = (px - pw / 2.0) / scale + dim_u / 2.0 - pan_x
        v = (py - ph / 2.0) / scale + dim_v / 2.0 - pan_y

        ui = int(ti.round(u))
        vi = int(ti.round(v))

        if 1 <= ui < dim_u - 1 and 1 <= vi < dim_v - 1:
            c = _sample_sdf(vol, axis, slice_idx, ui, vi)
            # Check if any neighbor has opposite sign → zero crossing
            is_contour = False
            r = _sample_sdf(vol, axis, slice_idx, ui + 1, vi)
            l = _sample_sdf(vol, axis, slice_idx, ui - 1, vi)
            t = _sample_sdf(vol, axis, slice_idx, ui, vi + 1)
            b = _sample_sdf(vol, axis, slice_idx, ui, vi - 1)

            if (c >= 0.0 and (r < 0.0 or l < 0.0 or t < 0.0 or b < 0.0)):
                is_contour = True
            if (c < 0.0 and (r >= 0.0 or l >= 0.0 or t >= 0.0 or b >= 0.0)):
                is_contour = True

            if is_contour:
                a = opacity
                bx = px0 + px
                by = py0 + py
                if 0 <= bx < bw and 0 <= by < bh:
                    old = buf[bx, by]
                    buf[bx, by] = old * (1.0 - a) + contour_color * a

"""Slice extraction and rendering kernels."""

import taichi as ti


@ti.func
def _get_slice_dims(vol: ti.template(), axis: int):
    """Return (dim_u, dim_v) for the two in-plane axes."""
    dim_u = vol.shape[0]
    dim_v = vol.shape[1]
    if axis == 2:  # axial: show (i, j) plane
        dim_u = vol.shape[0]
        dim_v = vol.shape[1]
    elif axis == 1:  # coronal: show (i, k) plane
        dim_u = vol.shape[0]
        dim_v = vol.shape[2]
    else:  # sagittal: show (j, k) plane
        dim_u = vol.shape[1]
        dim_v = vol.shape[2]
    return dim_u, dim_v


@ti.func
def _pixel_to_voxel(px: int, py: int, pw: int, ph: int,
                    dim_u: int, dim_v: int, zoom: float,
                    pan_x: float, pan_y: float):
    """Map panel pixel to voxel (u, v) coordinates."""
    scale = ti.min(float(pw) / float(dim_u), float(ph) / float(dim_v)) * zoom
    u = (px - pw / 2.0) / scale + dim_u / 2.0 - pan_x
    v = (py - ph / 2.0) / scale + dim_v / 2.0 - pan_y
    return u, v


@ti.func
def _sample_u8(vol: ti.template(), axis: int, slice_idx: int, ui: int, vi: int):
    """Sample a u8 volume given axis, slice index, and in-plane coords."""
    val = ti.cast(0, ti.u8)
    if axis == 2:
        val = vol[ui, vi, slice_idx]
    elif axis == 1:
        val = vol[ui, slice_idx, vi]
    else:
        val = vol[slice_idx, ui, vi]
    return val


@ti.func
def _sample_f32(vol: ti.template(), axis: int, slice_idx: int, ui: int, vi: int):
    """Sample a f32 volume given axis, slice index, and in-plane coords."""
    val = 0.0
    if axis == 2:
        val = vol[ui, vi, slice_idx]
    elif axis == 1:
        val = vol[ui, slice_idx, vi]
    else:
        val = vol[slice_idx, ui, vi]
    return val


@ti.kernel
def categorical_slice(
    vol: ti.template(),      # u8 3D field
    lut: ti.template(),      # vec4 LUT (256 entries)
    buf: ti.template(),      # vec3 composite buffer
    axis: int,               # 0=sagittal(i), 1=coronal(j), 2=axial(k)
    slice_idx: int,          # index along axis
    px0: int, py0: int,      # panel origin in buffer
    pw: int, ph: int,        # panel size
    opacity: float,          # layer opacity
    zoom: float,             # zoom factor
    pan_x: float, pan_y: float,  # pan offset in voxel coords
):
    """Render categorical volume slice into composite buffer with alpha blend."""
    for px, py in ti.ndrange(pw, ph):
        dim_u, dim_v = _get_slice_dims(vol, axis)
        u, v = _pixel_to_voxel(px, py, pw, ph, dim_u, dim_v, zoom, pan_x, pan_y)
        ui = int(ti.round(u))
        vi = int(ti.round(v))

        if 0 <= ui < dim_u and 0 <= vi < dim_v:
            label = _sample_u8(vol, axis, slice_idx, ui, vi)
            rgba = lut[int(label)]
            a = rgba.w * opacity
            if a > 0.0:
                bx = px0 + px
                by = py0 + py
                old = buf[bx, by]
                buf[bx, by] = old * (1.0 - a) + rgba.xyz * a


@ti.kernel
def scalar_slice(
    vol: ti.template(),      # f32 3D field
    lut: ti.template(),      # vec4 LUT (256 entries)
    buf: ti.template(),      # vec3 composite buffer
    axis: int,
    slice_idx: int,
    px0: int, py0: int,
    pw: int, ph: int,
    vmin: float, vmax: float,
    opacity: float,
    zoom: float,
    pan_x: float, pan_y: float,
):
    """Render scalar volume slice with continuous colormap."""
    for px, py in ti.ndrange(pw, ph):
        dim_u, dim_v = _get_slice_dims(vol, axis)
        u, v = _pixel_to_voxel(px, py, pw, ph, dim_u, dim_v, zoom, pan_x, pan_y)
        ui = int(ti.round(u))
        vi = int(ti.round(v))

        if 0 <= ui < dim_u and 0 <= vi < dim_v:
            val = _sample_f32(vol, axis, slice_idx, ui, vi)

            # Normalize to [0, 1] then look up in 256-entry LUT
            t = 0.0
            if vmax > vmin:
                t = (val - vmin) / (vmax - vmin)
            t = ti.max(0.0, ti.min(1.0, t))
            idx = int(t * 255.0)
            rgba = lut[idx]
            a = rgba.w * opacity
            if a > 0.0:
                bx = px0 + px
                by = py0 + py
                old = buf[bx, by]
                buf[bx, by] = old * (1.0 - a) + rgba.xyz * a

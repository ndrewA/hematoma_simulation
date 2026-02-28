"""Shared Taichi helper functions used by multiple kernels."""

import taichi as ti


@ti.func
def get_slice_dims(vol: ti.template(), axis: int):
    """Return (dim_u, dim_v) for the two in-plane axes."""
    dim_u = vol.shape[0]
    dim_v = vol.shape[1]
    if axis == 1:  # coronal: show (i, k) plane
        dim_u = vol.shape[0]
        dim_v = vol.shape[2]
    else:  # sagittal: show (j, k) plane
        dim_u = vol.shape[1]
        dim_v = vol.shape[2]
    return dim_u, dim_v


@ti.func
def pixel_to_voxel(px: int, py: int, pw: int, ph: int,
                   dim_u: int, dim_v: int, zoom: float,
                   pan_x: float, pan_y: float):
    """Map panel pixel to voxel (u, v) coordinates."""
    scale = ti.min(float(pw) / float(dim_u), float(ph) / float(dim_v)) * zoom
    u = (px - pw / 2.0) / scale + dim_u / 2.0 - pan_x
    v = (py - ph / 2.0) / scale + dim_v / 2.0 - pan_y
    return u, v


@ti.func
def slice_to_grid(axis: int, ui: float, vi: float, slice_idx: float):
    """Map in-plane coords (ui, vi) + slice index to 3D grid coords (gi, gj, gk)."""
    gi = 0.0
    gj = 0.0
    gk = 0.0
    if axis == 2:  # axial
        gi = ui
        gj = vi
        gk = slice_idx
    elif axis == 1:  # coronal
        gi = ui
        gj = slice_idx
        gk = vi
    else:  # sagittal
        gi = slice_idx
        gj = ui
        gk = vi
    return gi, gj, gk


@ti.func
def sample_f32(vol: ti.template(), axis: int, slice_idx: int,
               ui: int, vi: int) -> float:
    """Sample a f32 volume given axis, slice index, and in-plane coords."""
    val = 0.0
    if axis == 2:
        val = vol[ui, vi, slice_idx]
    elif axis == 1:
        val = vol[ui, slice_idx, vi]
    else:
        val = vol[slice_idx, ui, vi]
    return val

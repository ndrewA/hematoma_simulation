"""Shared Taichi helper functions used by multiple kernels."""

import taichi as ti


@ti.func
def get_slice_dims(vol: ti.template(), axis: int):
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

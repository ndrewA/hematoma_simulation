"""NIfTI data loading into Taichi fields."""

import json

import nibabel as nib
import numpy as np
import taichi as ti
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure

from preprocessing.utils import (
    PROFILES, processed_dir, raw_dir, build_grid_affine, resample_to_grid,
)


def _load_nifti(path):
    """Load a NIfTI file, return (data, affine)."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def load_volume_u8(path):
    """Load a NIfTI as uint8 Taichi field."""
    data, _ = _load_nifti(path)
    data = data.astype(np.uint8)
    field = ti.field(dtype=ti.u8, shape=data.shape)
    field.from_numpy(data)
    return field


def load_volume_f32(path):
    """Load a NIfTI as float32 Taichi field."""
    data, _ = _load_nifti(path)
    data = data.astype(np.float32)
    field = ti.field(dtype=ti.f32, shape=data.shape)
    field.from_numpy(data)
    return field


def load_t1w(subject_id, grid_affine, grid_shape):
    """Load T1w, resample to grid, return Taichi field + (vmin, vmax)."""
    t1w_path = raw_dir(subject_id) / "T1w_acpc_dc_restore.nii.gz"
    data = resample_to_grid(t1w_path, grid_affine, grid_shape, order=1,
                            dtype=np.float32)
    vmin, vmax = float(data.min()), float(np.percentile(data, 99.5))
    field = ti.field(dtype=ti.f32, shape=data.shape)
    field.from_numpy(data)
    return field, vmin, vmax


def load_fiber_tensor(path):
    """Load fiber M0 tensor (N,N,N,6) → Taichi vector field + affine + shape."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    spatial_shape = data.shape[:3]
    field = ti.Vector.field(6, dtype=ti.f32, shape=spatial_shape)
    field.from_numpy(data)
    return field, img.affine.copy(), spatial_shape


class ViewerData:
    """Container for all loaded volumetric data."""

    def __init__(self, subject_id, profile):
        self.subject_id = subject_id
        self.profile = profile

        pdir = processed_dir(subject_id, profile)
        self._load_grid_meta(pdir, profile)
        self._load_volumes(pdir, subject_id)
        self._derive_masks()
        self._load_fiber(pdir)

    def _load_grid_meta(self, pdir, profile):
        """Load grid metadata and determine actual grid geometry."""
        N, dx = PROFILES[profile]
        self.grid_affine = build_grid_affine(N, dx)

        with open(pdir / "grid_meta.json") as f:
            self.meta = json.load(f)

        sample = nib.load(str(pdir / "material_map.nii.gz"))
        self.grid_shape = sample.shape
        self.N = max(self.grid_shape)
        self.dx = dx

        if self.N != N:
            self.grid_affine = build_grid_affine(self.N, self.dx)

    def _load_volumes(self, pdir, subject_id):
        """Load all volumetric data into Taichi fields."""
        self.material_map = load_volume_u8(pdir / "material_map.nii.gz")
        self.skull_sdf = load_volume_f32(pdir / "skull_sdf.nii.gz")
        self.brain_mask = load_volume_u8(pdir / "brain_mask.nii.gz")
        self.t1w, self.t1w_vmin, self.t1w_vmax = load_t1w(
            subject_id, self.grid_affine, self.grid_shape)

    def _derive_masks(self):
        """Compute derived masks and centroids from loaded volumes."""
        mat_np = self.material_map.to_numpy()
        dura_np = (mat_np == 10).astype(np.uint8)
        self.dura_mask = ti.field(dtype=ti.u8, shape=self.grid_shape)
        self.dura_mask.from_numpy(dura_np)

        # Distance from each exterior voxel to skull+shell boundary (voxel units)
        # Dilate interior by 1 voxel so the inferred skull surface (SDF sign-
        # change voxels) is included as an obstacle for sphere tracing.
        sdf_np = self.skull_sdf.to_numpy()
        interior = sdf_np < 0
        struct6 = generate_binary_structure(3, 1)  # 6-connectivity
        obstacle = binary_dilation(interior, structure=struct6)
        exterior_dist = distance_transform_edt(~obstacle).astype(np.float32)
        self.exterior_dist = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.exterior_dist.from_numpy(exterior_dist)

        centroid = self.meta.get('brain_centroid_grid')
        if centroid:
            self.brain_centroid = tuple(int(round(c)) for c in centroid)
        else:
            self.brain_centroid = (self.N // 2, self.N // 2, self.N // 2)

    def _load_fiber(self, pdir):
        """Load fiber tensor data if available."""
        fiber_path = pdir.parent / "fiber_M0.nii.gz"
        self.fiber_field = None
        self.fiber_affine = None
        self.fiber_shape = None
        if fiber_path.exists():
            self.fiber_field, self.fiber_affine, self.fiber_shape = \
                load_fiber_tensor(fiber_path)

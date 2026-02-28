"""NIfTI data loading into Taichi fields."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import taichi as ti

# Reuse path helpers from preprocessing
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
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
        N, dx = PROFILES[profile]
        grid_affine = build_grid_affine(N, dx)

        # Load grid metadata
        meta_path = pdir / "grid_meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        # Actual grid shape from the data (may differ from PROFILES if
        # data was generated with older settings)
        sample = nib.load(str(pdir / "material_map.nii.gz"))
        self.grid_shape = sample.shape
        # Use max dimension for cubic assumption (viewer assumes cubic N)
        self.N = max(self.grid_shape)
        self.dx = dx
        self.grid_affine = grid_affine

        # If actual data shape doesn't match PROFILES, rebuild affine
        if self.N != N:
            self.grid_affine = build_grid_affine(self.N, self.dx)

        # Material map (categorical)
        self.material_map = load_volume_u8(pdir / "material_map.nii.gz")

        # Skull SDF (float)
        self.skull_sdf = load_volume_f32(pdir / "skull_sdf.nii.gz")

        # Brain mask (u8)
        self.brain_mask = load_volume_u8(pdir / "brain_mask.nii.gz")

        # T1w (resampled)
        self.t1w, self.t1w_vmin, self.t1w_vmax = load_t1w(
            subject_id, self.grid_affine, self.grid_shape)

        # Dura mask (label 10 from material map)
        mat_np = self.material_map.to_numpy()
        dura_np = (mat_np == 10).astype(np.uint8)
        self.dura_mask = ti.field(dtype=ti.u8, shape=self.grid_shape)
        self.dura_mask.from_numpy(dura_np)

        # Brain centroid for camera/crosshair centering
        centroid = self.meta.get('brain_centroid_grid')
        if centroid:
            self.brain_centroid = tuple(int(round(c)) for c in centroid)
        else:
            self.brain_centroid = (self.N // 2, self.N // 2, self.N // 2)

        # Fiber tensor (native resolution, shared across profiles)
        fiber_path = pdir.parent / "fiber_M0.nii.gz"
        self.fiber_field = None
        self.fiber_affine = None
        self.fiber_shape = None
        if fiber_path.exists():
            self.fiber_field, self.fiber_affine, self.fiber_shape = \
                load_fiber_tensor(fiber_path)

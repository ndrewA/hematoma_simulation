"""Colormap LUT construction for Taichi fields."""

import taichi as ti
import numpy as np

# Categorical colors from preprocessing/validation/figures.py
MATERIAL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),    # 0: Vacuum (transparent)
    (1.0, 1.0, 1.0, 1.0),    # 1: Cerebral WM (white)
    (0.3, 0.3, 0.3, 1.0),    # 2: Cortical GM (dark gray)
    (1.0, 1.0, 0.0, 1.0),    # 3: Deep GM (yellow)
    (1.0, 1.0, 0.94, 1.0),   # 4: Cerebellar WM (ivory)
    (0.5, 0.5, 0.0, 1.0),    # 5: Cerebellar Cortex (olive)
    (1.0, 0.65, 0.0, 1.0),   # 6: Brainstem (orange)
    (0.0, 0.0, 1.0, 1.0),    # 7: Ventricular CSF (blue)
    (0.0, 1.0, 1.0, 1.0),    # 8: Subarachnoid CSF (cyan)
    (1.0, 0.0, 1.0, 1.0),    # 9: Choroid Plexus (magenta)
    (1.0, 0.0, 0.0, 1.0),    # 10: Dural Membrane (red)
    (0.0, 1.0, 0.0, 1.0),    # 11: Vessel / Venous Sinus (green)
    (0.92, 0.87, 0.78, 1.0),  # 12: Skull (bone beige) — virtual label for 3D
]

CLASS_NAMES = {
    0: "Vacuum",
    1: "Cerebral WM",
    2: "Cortical GM",
    3: "Deep GM",
    4: "Cerebellar WM",
    5: "Cerebellar Cortex",
    6: "Brainstem",
    7: "Ventricular CSF",
    8: "Subarachnoid CSF",
    9: "Choroid Plexus",
    10: "Dural Membrane",
    11: "Vessel / Venous Sinus",
    12: "Skull",
}

N_CATEGORIES = 256  # max u8 value + 1


def build_categorical_lut():
    """Build a 256-entry RGBA LUT for categorical labels as a Taichi field."""
    lut = ti.Vector.field(4, dtype=ti.f32, shape=(N_CATEGORIES,))
    arr = np.zeros((N_CATEGORIES, 4), dtype=np.float32)
    for i, rgba in enumerate(MATERIAL_COLORS):
        arr[i] = rgba
    lut.from_numpy(arr)
    return lut


def build_grayscale_lut():
    """Build a 256-entry grayscale LUT."""
    lut = ti.Vector.field(4, dtype=ti.f32, shape=(256,))
    arr = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        v = i / 255.0
        arr[i] = (v, v, v, 1.0)
    lut.from_numpy(arr)
    return lut


def build_highlight_lut(r, g, b):
    """Build a 256-entry LUT where 0=transparent, 1=(r,g,b,1)."""
    lut = ti.Vector.field(4, dtype=ti.f32, shape=(N_CATEGORIES,))
    arr = np.zeros((N_CATEGORIES, 4), dtype=np.float32)
    arr[1] = (r, g, b, 1.0)
    lut.from_numpy(arr)
    return lut


def build_diverging_lut():
    """Build a 256-entry blue-white-red diverging LUT for SDF."""
    lut = ti.Vector.field(4, dtype=ti.f32, shape=(256,))
    arr = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        t = i / 255.0  # 0 = most negative, 0.5 = zero, 1.0 = most positive
        if t < 0.5:
            s = t / 0.5  # 0..1
            arr[i] = (s, s, 1.0, 1.0)  # blue → white
        else:
            s = (t - 0.5) / 0.5  # 0..1
            arr[i] = (1.0, 1.0 - s, 1.0 - s, 1.0)  # white → red
    lut.from_numpy(arr)
    return lut

"""Layer definitions for the viewer."""

from dataclasses import dataclass, field
from enum import Enum, auto

import taichi as ti


class LayerType(Enum):
    CATEGORICAL = auto()
    SCALAR = auto()
    TENSOR = auto()
    CONTOUR = auto()  # draws zero-crossing of a signed field


@dataclass
class Layer:
    name: str
    layer_type: LayerType
    field: ti.Field
    visible: bool = True
    opacity: float = 1.0
    vmin: float = 0.0
    vmax: float = 1.0
    lut_index: int = 0  # index into the LUT array for this layer's colormap

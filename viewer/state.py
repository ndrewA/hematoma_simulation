"""View state, camera, and layout management."""

import math
from dataclasses import dataclass
from enum import IntEnum


@dataclass
class PanelRect:
    x0: int
    y0: int
    w: int
    h: int


@dataclass
class PanelView:
    """Per-panel view state (zoom + pan)."""
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0


class DragType(IntEnum):
    NONE = 0
    ORBIT = 1
    SLICE_PAN = 2
    PLANE_DRAG = 3


class DragState:
    """Encapsulates drag interaction state."""

    def __init__(self):
        self.reset()

    def start(self, drag_type, mouse, lmb=True, panel=-1, axis=-1, pos=0.0):
        self.type = drag_type
        self.prev = mouse
        self.lmb = lmb
        self.panel = panel
        self.axis = axis
        self.pos = pos

    def reset(self):
        self.type = DragType.NONE
        self.prev = None
        self.panel = -1
        self.axis = -1
        self.pos = 0.0
        self.lmb = True

    @property
    def active(self):
        return self.type != DragType.NONE


class LayoutMode(IntEnum):
    SLICES = 0     # 3 slice panels
    THREE_D = 1    # 3D fullscreen


class OrbitalCamera:
    """Orbital camera in RAS+ coordinates (Z-up = Superior)."""

    def __init__(self):
        self.azimuth = 0.4        # radians in XY plane
        self.elevation = 0.4      # radians up from XY plane
        self.distance = 1.0       # will be set based on N
        self.fov_deg = 40.0
        self.center = (0.0, 0.0, 0.0)

    def eye_position(self):
        r = self.distance
        ce = math.cos(self.elevation)
        se = math.sin(self.elevation)
        ca = math.cos(self.azimuth)
        sa = math.sin(self.azimuth)
        x = r * ce * ca + self.center[0]
        y = r * ce * sa + self.center[1]
        z = r * se + self.center[2]
        return (x, y, z)

    def basis(self):
        """Return (fwd, right, up) basis vectors for the camera."""
        eye = self.eye_position()
        c = self.center
        fwd = (c[0] - eye[0], c[1] - eye[1], c[2] - eye[2])
        length = math.sqrt(sum(v*v for v in fwd))
        if length < 1e-6:
            return None
        fwd = tuple(v / length for v in fwd)

        world_up = (0.0, 0.0, 1.0)  # RAS+: Z = Superior = up
        right = (
            fwd[1] * world_up[2] - fwd[2] * world_up[1],
            fwd[2] * world_up[0] - fwd[0] * world_up[2],
            fwd[0] * world_up[1] - fwd[1] * world_up[0],
        )
        rl = math.sqrt(sum(v*v for v in right))
        if rl < 1e-6:
            return None
        right = tuple(v / rl for v in right)

        up = (
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        )
        return fwd, right, up


class LayoutManager:
    """Computes panel rectangles based on layout mode."""

    SIDEBAR_W = 180  # pixels reserved for sidebar

    def __init__(self):
        self.panels = [PanelRect(0, 0, 0, 0) for _ in range(4)]
        self.sidebar_x0 = 0  # pixel x where sidebar starts

    def update(self, win_w, win_h, mode, fullscreen_panel):
        # Reset all
        for i in range(4):
            self.panels[i] = PanelRect(0, 0, 0, 0)

        if fullscreen_panel >= 0:
            self.panels[fullscreen_panel] = PanelRect(0, 0, win_w, win_h)
            self.sidebar_x0 = win_w
        elif mode == LayoutMode.SLICES:
            gap = 2
            cw = win_w - self.SIDEBAR_W  # content width
            self.sidebar_x0 = cw
            pw = (cw - gap) // 2
            hh = (win_h - gap) // 2
            # Top row: axial left, coronal right
            self.panels[0] = PanelRect(0, hh + gap, pw, hh)
            self.panels[1] = PanelRect(pw + gap, hh + gap, pw, hh)
            # Bottom row: sagittal left, 3D crosshair widget right
            self.panels[2] = PanelRect(0, 0, pw, hh)
            self.panels[3] = PanelRect(pw + gap, 0, pw, hh)
        elif mode == LayoutMode.THREE_D:
            cw = win_w - self.SIDEBAR_W
            self.sidebar_x0 = cw
            self.panels[3] = PanelRect(0, 0, cw, win_h)

    def hit_test(self, mx, my):
        for i, p in enumerate(self.panels):
            if p.w > 0 and p.h > 0:
                if p.x0 <= mx < p.x0 + p.w and p.y0 <= my < p.y0 + p.h:
                    return i
        return -1


class ViewState:
    """Central mutable state for the viewer."""

    def __init__(self, N):
        self.N = N
        self.crosshair = [N // 2, N // 2, N // 2]
        self.fullscreen_panel = -1
        self.layout_mode = LayoutMode.SLICES
        self.panel_views = [PanelView() for _ in range(4)]
        self.camera = OrbitalCamera()
        self.camera.distance = N * 1.0
        self.camera.center = (N / 2.0, N / 2.0, N / 2.0)
        self.layers = []
        self.layout = LayoutManager()
        self.focused_panel = -1
        self.show_ui = True
        # 3D voxel renderer: per-group opacity [brain, csf, dura, skull, choroid, vessel]
        self.group_opacity = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        self.drag = DragState()

    PANEL_AXIS = [2, 1, 0]  # panel 0=axial(k), 1=coronal(j), 2=sagittal(i)
    # Maps axis → (crosshair_u_index, crosshair_v_index) for in-plane coords
    AXIS_UV = {2: (0, 1), 1: (0, 2), 0: (1, 2)}

    def slice_index(self, panel):
        return self.crosshair[self.PANEL_AXIS[panel]]

    def set_slice_index(self, panel, val):
        axis = self.PANEL_AXIS[panel]
        self.crosshair[axis] = max(0, min(self.N - 1, val))

    def widget_camera(self):
        """Camera for the 3D crosshair widget — centered on cube, framed to fit."""
        cam = OrbitalCamera()
        cam.azimuth = self.camera.azimuth
        cam.elevation = self.camera.elevation
        cam.fov_deg = self.camera.fov_deg
        cam.center = (self.N / 2.0, self.N / 2.0, self.N / 2.0)
        cam.distance = self.N * 2.0
        return cam

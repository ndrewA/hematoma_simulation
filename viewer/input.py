"""Input handling: mouse/keyboard → state mutations."""

import math
import taichi as ti
from viewer.state import LayoutMode


_prev_keys = set()


def process_input(window, state):
    """Process input events and update ViewState. Returns False to quit."""
    global _prev_keys

    curr_keys = set()
    consumed = set()

    def pressed(key):
        is_down = window.is_pressed(key)
        if is_down:
            curr_keys.add(key)
        if is_down and key not in _prev_keys and key not in consumed:
            consumed.add(key)
            return True
        return False

    # --- Quit ---
    if pressed(ti.ui.ESCAPE):
        if state.fullscreen_panel >= 0:
            state.fullscreen_panel = -1
        else:
            _prev_keys = curr_keys
            return False

    # --- Mouse ---
    mouse = window.get_cursor_pos()
    win_w, win_h = window.get_window_shape()
    mx = int(mouse[0] * win_w)
    my = int(mouse[1] * win_h)
    hovered = state.layout.hit_test(mx, my)
    if hovered >= 0:
        state.focused_panel = hovered

    # --- Mouse drag for 3D orbit ---
    lmb = window.is_pressed(ti.ui.LMB)
    if hovered == 3 and lmb:
        if state._prev_mouse is not None:
            dx = mouse[0] - state._prev_mouse[0]
            dy = mouse[1] - state._prev_mouse[1]
            state.camera.azimuth -= dx * 3.0
            state.camera.elevation -= dy * 3.0
            state.camera.elevation = max(-1.4, min(1.4, state.camera.elevation))
            state._dragging = True
        state._prev_mouse = mouse
    else:
        # Click to set crosshair (only if not dragging, not on GUI)
        if lmb and 0 <= hovered <= 2 and not state._dragging:
            _set_crosshair_from_click(state, hovered, mx, my)
        state._prev_mouse = None
        state._dragging = False

    # --- Layout mode: Tab toggles Slices ↔ 3D ---
    if pressed(ti.ui.TAB):
        state.layout_mode = LayoutMode((state.layout_mode + 1) % 2)
        state.fullscreen_panel = -1

    # --- Fullscreen toggle ---
    if pressed(ti.ui.RETURN):
        if state.fullscreen_panel >= 0:
            state.fullscreen_panel = -1
        elif state.focused_panel >= 0:
            state.fullscreen_panel = state.focused_panel

    # --- Slice navigation ---
    fp = state.focused_panel
    if 0 <= fp <= 2:
        if pressed(ti.ui.UP):
            state.set_slice_index(fp, state.slice_index(fp) + 1)
        if pressed(ti.ui.DOWN):
            state.set_slice_index(fp, state.slice_index(fp) - 1)

    # --- Zoom (slices) ---
    if pressed('='):
        if 0 <= fp <= 2:
            state.panel_zoom[fp] = min(state.panel_zoom[fp] * 1.2, 20.0)
    if pressed('-'):
        if 0 <= fp <= 2:
            state.panel_zoom[fp] = max(state.panel_zoom[fp] / 1.2, 0.1)

    # --- 3D camera: WASD orbit + zoom ---
    if fp == 3 or state.layout_mode == LayoutMode.THREE_D:
        cam = state.camera
        speed = 0.06
        if window.is_pressed('a'):
            cam.azimuth += speed
        if window.is_pressed('d'):
            cam.azimuth -= speed
        if window.is_pressed('w'):
            cam.elevation = min(cam.elevation + speed, 1.4)
        if window.is_pressed('s'):
            cam.elevation = max(cam.elevation - speed, -1.4)
        # Zoom 3D with +/-
        if pressed('=') or pressed(ti.ui.UP):
            cam.distance = max(cam.distance * 0.85, 10.0)
        if pressed('-') or pressed(ti.ui.DOWN):
            cam.distance = cam.distance * 1.18

    # --- UI toggle ---
    if pressed('h'):
        state.show_ui = not state.show_ui

    # --- Layer toggles (1-6) ---
    for idx, key in enumerate('123456'):
        if pressed(key) and idx < len(state.layers):
            state.layers[idx].visible = not state.layers[idx].visible

    _prev_keys = curr_keys
    return True


def _set_crosshair_from_click(state, panel, mx, my):
    p = state.layout.panels[panel]
    if p.w == 0 or p.h == 0:
        return

    axis = state.PANEL_AXIS[panel]
    N = state.N
    dim_u, dim_v = N, N

    zoom = state.panel_zoom[panel]
    pan_x, pan_y = state.panel_pan[panel]
    scale = min(p.w / dim_u, p.h / dim_v) * zoom

    lx = mx - p.x0
    ly = my - p.y0
    u = (lx - p.w / 2.0) / scale + dim_u / 2.0 - pan_x
    v = (ly - p.h / 2.0) / scale + dim_v / 2.0 - pan_y
    ui = int(round(u))
    vi = int(round(v))

    if 0 <= ui < dim_u and 0 <= vi < dim_v:
        if axis == 2:
            state.crosshair[0] = ui
            state.crosshair[1] = vi
        elif axis == 1:
            state.crosshair[0] = ui
            state.crosshair[2] = vi
        else:
            state.crosshair[1] = ui
            state.crosshair[2] = vi

"""Input handling: mouse/keyboard → state mutations."""

import math
import taichi as ti
from viewer.state import DragType, LayoutMode


def process_input(window, state):
    """Process input events and update ViewState. Returns False to quit."""
    key_presses = _collect_events(window)

    if ti.ui.ESCAPE in key_presses:
        if state.fullscreen_panel >= 0:
            state.fullscreen_panel = -1
        else:
            return False

    mouse = window.get_cursor_pos()
    win_w, win_h = window.get_window_shape()
    mx = int(mouse[0] * win_w)
    my = int(mouse[1] * win_h)
    hovered = state.layout.hit_test(mx, my)
    if hovered >= 0:
        state.focused_panel = hovered

    _handle_mouse(window, state, mouse, hovered, mx, my, win_w, win_h)
    _handle_keys(window, state, key_presses)
    return True


def _collect_events(window):
    """Drain discrete key-press events for this frame."""
    key_presses = set()
    while window.get_event(ti.ui.PRESS):
        key_presses.add(window.event.key)
    return key_presses


# --- Drag dispatch ---

def _apply_drag(state, mouse, win_w, win_h):
    drag = state.drag
    prev = drag.prev
    if prev is None:
        drag.prev = mouse
        return

    if drag.type == DragType.ORBIT:
        dx = mouse[0] - prev[0]
        dy = mouse[1] - prev[1]
        state.camera.azimuth -= dx * 3.0
        state.camera.elevation -= dy * 3.0
        state.camera.elevation = max(-1.4, min(1.4, state.camera.elevation))

    elif drag.type == DragType.SLICE_PAN:
        panel = drag.panel
        p = state.layout.panels[panel]
        N = state.N
        zoom = state.panel_zoom[panel]
        scale = min(p.w / N, p.h / N) * zoom
        dx_px = (mouse[0] - prev[0]) * win_w
        dy_px = (mouse[1] - prev[1]) * win_h
        pan_x, pan_y = state.panel_pan[panel]
        state.panel_pan[panel] = (pan_x + dx_px / scale, pan_y + dy_px / scale)

    elif drag.type == DragType.PLANE_DRAG:
        _apply_plane_drag(state, mouse, prev, state.widget_camera(), win_w, win_h)

    drag.prev = mouse


# --- Mouse handling ---

def _handle_mouse(window, state, mouse, hovered, mx, my, win_w, win_h):
    lmb = window.is_pressed(ti.ui.LMB)
    rmb = window.is_pressed(ti.ui.RMB)

    # Active drag: continue or end
    if state.drag.active:
        btn_held = lmb if state.drag.lmb else rmb
        if btn_held:
            _apply_drag(state, mouse, win_w, win_h)
        else:
            state.drag.reset()
        return

    # No active drag — check for new interactions
    if lmb:
        if hovered == 3 and state.layout_mode == LayoutMode.SLICES:
            axis = _pick_plane(state, mx, my, state.widget_camera())
            if axis >= 0:
                state.drag.start(DragType.PLANE_DRAG, mouse, lmb=True,
                                 axis=axis, pos=float(state.crosshair[axis]))
        elif hovered == 3:
            state.drag.start(DragType.ORBIT, mouse, lmb=True)
        elif 0 <= hovered <= 2:
            _set_crosshair_from_click(state, hovered, mx, my)
        return

    if rmb:
        if hovered == 3:
            state.drag.start(DragType.ORBIT, mouse, lmb=False)
        elif 0 <= hovered <= 2:
            state.drag.start(DragType.SLICE_PAN, mouse, lmb=False, panel=hovered)


# --- Key handling ---

def _handle_keys(window, state, key_presses):
    # Layout mode: Tab toggles Slices ↔ 3D
    if ti.ui.TAB in key_presses:
        state.layout_mode = LayoutMode((state.layout_mode + 1) % 2)
        state.fullscreen_panel = -1

    # Fullscreen toggle
    if ti.ui.RETURN in key_presses:
        if state.fullscreen_panel >= 0:
            state.fullscreen_panel = -1
        elif state.focused_panel >= 0:
            state.fullscreen_panel = state.focused_panel

    # Slice navigation and zoom vs 3D camera
    fp = state.focused_panel
    in_3d_context = (fp == 3 or
                     (state.layout_mode == LayoutMode.THREE_D and not (0 <= fp <= 2)))
    if in_3d_context:
        _keys_3d(window, state, key_presses)
    elif 0 <= fp <= 2:
        _keys_slice(state, fp, key_presses)

    # UI toggle
    if 'h' in key_presses:
        state.show_ui = not state.show_ui

    # Layer toggles (F/G/Z/X/C/V)
    layer_keys = 'fgzxcv'
    for idx, key in enumerate(layer_keys):
        if key in key_presses and idx < len(state.layers):
            state.layers[idx].visible = not state.layers[idx].visible


def _keys_3d(window, state, key_presses):
    cam = state.camera
    speed = 0.06
    if window.is_pressed('a'):
        cam.azimuth -= speed
    if window.is_pressed('d'):
        cam.azimuth += speed
    if window.is_pressed('w'):
        cam.elevation = min(cam.elevation + speed, 1.4)
    if window.is_pressed('s'):
        cam.elevation = max(cam.elevation - speed, -1.4)
    if 'e' in key_presses or ti.ui.UP in key_presses:
        cam.distance = max(cam.distance * 0.85, 10.0)
    if 'q' in key_presses or ti.ui.DOWN in key_presses:
        cam.distance = cam.distance * 1.18


def _keys_slice(state, fp, key_presses):
    if ti.ui.UP in key_presses:
        state.set_slice_index(fp, state.slice_index(fp) + 1)
    if ti.ui.DOWN in key_presses:
        state.set_slice_index(fp, state.slice_index(fp) - 1)
    if 'e' in key_presses:
        state.panel_zoom[fp] = min(state.panel_zoom[fp] * 1.2, 20.0)
    if 'q' in key_presses:
        state.panel_zoom[fp] = max(state.panel_zoom[fp] / 1.2, 0.1)
    if 'r' in key_presses:
        state.panel_pan[fp] = (0.0, 0.0)
        state.panel_zoom[fp] = 1.0


# --- Crosshair and plane helpers ---

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
        cu, cv = state.AXIS_UV[axis]
        state.crosshair[cu] = ui
        state.crosshair[cv] = vi


def _pick_plane(state, mx, my, cam):
    """Ray-cast from mouse into 3D widget to find nearest crosshair plane."""
    p = state.layout.panels[3]
    if p.w == 0 or p.h == 0:
        return -1

    basis = cam.basis()
    if basis is None:
        return -1
    fwd, right, up = basis
    eye = cam.eye_position()
    fov_scale = math.tan(math.radians(cam.fov_deg) / 2.0)

    # Pixel position relative to panel
    px = mx - p.x0
    py = my - p.y0
    u_ndc = (2.0 * (px + 0.5) / p.w - 1.0) * fov_scale * p.w / p.h
    v_ndc = (2.0 * (py + 0.5) / p.h - 1.0) * fov_scale

    rd = [fwd[i] + right[i] * u_ndc + up[i] * v_ndc for i in range(3)]
    rd_len = math.sqrt(sum(v * v for v in rd))
    rd = [v / rd_len for v in rd]

    N = state.N
    best_axis = -1
    best_t = 1e30

    for axis in range(3):
        if abs(rd[axis]) < 1e-8:
            continue
        t = (state.crosshair[axis] - eye[axis]) / rd[axis]
        if t <= 0:
            continue
        # Check if hit is inside [0, N] box
        hit = [eye[i] + rd[i] * t for i in range(3)]
        inside = all(0 <= hit[i] <= N for i in range(3))
        if inside and t < best_t:
            best_t = t
            best_axis = axis

    return best_axis


def _apply_plane_drag(state, mouse, prev_mouse, cam, win_w, win_h):
    """Move the grabbed crosshair plane based on mouse delta."""
    axis = state.drag.axis
    basis = cam.basis()
    if basis is None:
        return
    fwd, right, up = basis
    fov_scale = math.tan(math.radians(cam.fov_deg) / 2.0)

    # Axis unit vector
    axis_dir = [0.0, 0.0, 0.0]
    axis_dir[axis] = 1.0

    # Project axis direction onto screen (right, up)
    screen_x = sum(axis_dir[i] * right[i] for i in range(3))
    screen_y = sum(axis_dir[i] * up[i] for i in range(3))
    screen_len = math.sqrt(screen_x ** 2 + screen_y ** 2)
    if screen_len < 1e-6:
        return

    # Mouse delta in pixels
    dx_px = (mouse[0] - prev_mouse[0]) * win_w
    dy_px = (mouse[1] - prev_mouse[1]) * win_h

    # Project mouse delta onto axis screen direction
    proj_px = (dx_px * screen_x + dy_px * screen_y) / screen_len

    # Convert pixel displacement to voxel displacement
    p = state.layout.panels[3]
    half_h = p.h / 2.0
    voxels_per_pixel = (cam.distance * fov_scale) / half_h
    delta = proj_px * voxels_per_pixel

    state.drag.pos += delta
    clamped = max(0, min(state.N - 1, int(round(state.drag.pos))))
    state.crosshair[axis] = clamped

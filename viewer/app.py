"""Main application: window, render loop, dispatch."""

import math
import numpy as np
import taichi as ti
from viewer.state import ViewState, LayoutMode
from viewer.data import ViewerData
from viewer.layers import Layer, LayerType
from viewer.colormaps import (
    build_categorical_lut, build_grayscale_lut, build_diverging_lut,
    build_highlight_lut,
)
from viewer.kernels.composite import clear, fill_rect, draw_crosshair, draw_panel_border
from viewer.kernels.slice_render import categorical_slice, scalar_slice
from viewer.kernels.voxel_trace import voxel_trace, N_GROUPS
from viewer.kernels.dec_render import dec_slice
from viewer.kernels.contour_render import contour_slice
from viewer.kernels.crosshair_3d import crosshair_3d
from viewer.input import process_input


@ti.kernel
def blit(src: ti.template(), dst: ti.template(), w: int, h: int):
    for x, y in ti.ndrange(w, h):
        dst[x, y] = src[x, y]


LAYOUT_NAMES = {LayoutMode.SLICES: "Slices", LayoutMode.THREE_D: "3D"}


def launch(subject_id, profile):
    ti.init(arch=ti.vulkan)

    print(f"Loading {subject_id}/{profile}...")
    data = ViewerData(subject_id, profile)
    N = data.N
    print(f"Grid: {N}³, dx={data.dx}mm")

    # LUTs
    cat_lut = build_categorical_lut()
    gray_lut = build_grayscale_lut()
    div_lut = build_diverging_lut()
    dura_lut = build_highlight_lut(1.0, 0.2, 0.2)  # red highlight for dura

    # Layers
    layers = [
        Layer("T1w", LayerType.SCALAR, data.t1w,
              visible=True, opacity=1.0,
              vmin=data.t1w_vmin, vmax=data.t1w_vmax),
        Layer("Material Map", LayerType.CATEGORICAL, data.material_map,
              visible=False, opacity=0.7),
        Layer("Skull Contour", LayerType.CONTOUR, data.skull_sdf,
              visible=False, opacity=1.0),
        Layer("Brain Mask", LayerType.CATEGORICAL, data.brain_mask,
              visible=False, opacity=0.3),
        Layer("Dura", LayerType.CATEGORICAL, data.dura_mask,
              visible=False, opacity=0.8),
    ]

    g2f_matrix = None
    if data.fiber_field is not None:
        layers.append(Layer("Fiber DEC", LayerType.TENSOR, data.fiber_field,
                            visible=False, opacity=0.8))
        fiber_affine_inv = np.linalg.inv(data.fiber_affine)
        g2f_matrix = (fiber_affine_inv @ data.grid_affine)[:3, :]

    luts = [cat_lut, gray_lut, div_lut, dura_lut]
    layers[0].lut_index = 1  # T1w → grayscale
    layers[1].lut_index = 0  # material → categorical
    # layers[2] is CONTOUR — no LUT needed
    layers[3].lut_index = 0  # brain mask → categorical
    layers[4].lut_index = 3  # dura → red highlight

    # Group opacity field for 3D voxel renderer
    group_opacity = ti.field(dtype=ti.f32, shape=(N_GROUPS,))
    for i in range(N_GROUPS):
        group_opacity[i] = 1.0

    # State
    state = ViewState(N)
    state.layers = layers
    cx, cy, cz = data.brain_centroid
    state.crosshair = [cx, cy, cz]
    state.camera.center = (float(cx), float(cy), float(cz))

    # Window
    init_w, init_h = 1280, 720
    window = ti.ui.Window("Viewer", (init_w, init_h), vsync=True)
    canvas = window.get_canvas()
    buf = ti.Vector.field(3, dtype=ti.f32, shape=(init_w, init_h))
    display = ti.Vector.field(3, dtype=ti.f32, shape=(init_w, init_h))
    buf_size = [init_w, init_h]

    print("Controls:")
    layer_keys = 'FGZXCV'
    print("  Left-click slice: crosshair  Right-drag slice: pan  R: reset view")
    print("  Up/Down: scroll slices       E/Q: zoom in/out     Tab: layout mode")
    print("  Drag 3D: orbit   WASD: orbit  H: toggle UI  Enter: fullscreen  Esc: quit")
    print(f"  Layers: {', '.join(f'{layer_keys[i]}={l.name}' for i, l in enumerate(layers))}")

    # Warmup: JIT-compile voxel_trace kernel before user interaction
    print("Compiling 3D kernel...", end="", flush=True)
    voxel_trace(
        data.material_map, data.skull_sdf, data.exterior_dist,
        cat_lut, group_opacity, buf,
        0, 0, 1, 1,  # 1x1 pixel
        0.0, 0.0, float(N) * 2.0,  # eye far away
        0.0, 1.0, 0.0,  # fwd
        1.0, 0.0, 0.0,  # right
        0.0, 0.0, 1.0,  # up
        0.36, N,
    )
    print(" done")

    while window.running:
        win_w, win_h = window.get_window_shape()

        if win_w != buf_size[0] or win_h != buf_size[1]:
            buf.destroy()
            buf = ti.Vector.field(3, dtype=ti.f32, shape=(win_w, win_h))
            display.destroy()
            display = ti.Vector.field(3, dtype=ti.f32, shape=(win_w, win_h))
            buf_size = [win_w, win_h]

        state.layout.update(win_w, win_h, state.layout_mode, state.fullscreen_panel)

        if not process_input(window, state):
            break

        clear(buf, win_w, win_h, 0.08, 0.08, 0.10)

        # Sidebar background (slightly lighter than main bg)
        sb_x0 = state.layout.sidebar_x0
        if sb_x0 < win_w:
            fill_rect(buf, sb_x0, 0, win_w - sb_x0, win_h, 0.11, 0.11, 0.13)

        _render_slice_panels(state, layers, luts, g2f_matrix, buf, N)

        # Render 3D panel (voxel trace in 3D mode, crosshair widget in slice mode)
        p3 = state.layout.panels[3]
        if p3.w > 0 and p3.h > 0:
            if state.layout_mode == LayoutMode.THREE_D:
                for gi in range(N_GROUPS):
                    group_opacity[gi] = state.group_opacity[gi]
                _render_3d(state, data, buf, p3, cat_lut, group_opacity)
            else:
                _render_crosshair_widget(state, buf, p3)
            focused = (3 == state.focused_panel)
            draw_panel_border(buf, p3.x0, p3.y0, p3.w, p3.h,
                              *(0.6, 0.6, 0.2) if focused else (0.15, 0.15, 0.18))

        blit(buf, display, win_w, win_h)
        canvas.set_image(display)

        if state.show_ui:
            _draw_gui(window, state, layers)

        window.show()

    window.destroy()


def _draw_panel_crosshair(state, panel_idx, buf):
    p = state.layout.panels[panel_idx]
    axis = state.PANEL_AXIS[panel_idx]
    N = state.N
    pv = state.panel_views[panel_idx]

    ui, vi = state.AXIS_UV[axis]
    cu_vox, cv_vox = state.crosshair[ui], state.crosshair[vi]

    scale = min(p.w / N, p.h / N) * pv.zoom
    cx = int((cu_vox - N / 2.0 + pv.pan_x) * scale + p.w / 2.0)
    cy = int((cv_vox - N / 2.0 + pv.pan_y) * scale + p.h / 2.0)

    if 0 <= cx < p.w and 0 <= cy < p.h:
        draw_crosshair(buf, p.x0, p.y0, p.w, p.h, cx, cy)


def _render_layer(layer, luts, g2f_matrix, buf, axis, si, p, zoom, pan_x, pan_y, N):
    """Dispatch a single layer to the appropriate slice kernel."""
    if layer.layer_type == LayerType.CATEGORICAL:
        lut = luts[layer.lut_index]
        categorical_slice(
            layer.field, lut, buf, axis, si,
            p.x0, p.y0, p.w, p.h,
            layer.opacity, zoom, pan_x, pan_y)
    elif layer.layer_type == LayerType.SCALAR:
        lut = luts[layer.lut_index]
        scalar_slice(
            layer.field, lut, buf, axis, si,
            p.x0, p.y0, p.w, p.h,
            layer.vmin, layer.vmax,
            layer.opacity, zoom, pan_x, pan_y)
    elif layer.layer_type == LayerType.CONTOUR:
        contour_slice(
            layer.field, buf, axis, si,
            p.x0, p.y0, p.w, p.h,
            layer.opacity, zoom, pan_x, pan_y,
            1.0, 0.2, 0.2)
    elif layer.layer_type == LayerType.TENSOR and g2f_matrix is not None:
        M = g2f_matrix
        dec_slice(
            layer.field, buf, axis, si,
            p.x0, p.y0, p.w, p.h,
            layer.opacity, zoom, pan_x, pan_y,
            M[0,0], M[0,1], M[0,2], M[0,3],
            M[1,0], M[1,1], M[1,2], M[1,3],
            M[2,0], M[2,1], M[2,2], M[2,3], N)


def _render_slice_panels(state, layers, luts, g2f_matrix, buf, N):
    """Render all slice panels: layers + crosshair + border per panel."""
    for panel_idx in range(3):
        p = state.layout.panels[panel_idx]
        if p.w == 0 or p.h == 0:
            continue

        axis = state.PANEL_AXIS[panel_idx]
        si = state.slice_index(panel_idx)
        pv = state.panel_views[panel_idx]

        for layer in layers:
            if layer.visible:
                _render_layer(layer, luts, g2f_matrix, buf,
                              axis, si, p, pv.zoom, pv.pan_x, pv.pan_y, N)

        _draw_panel_crosshair(state, panel_idx, buf)
        focused = (panel_idx == state.focused_panel)
        draw_panel_border(buf, p.x0, p.y0, p.w, p.h,
                          *(0.6, 0.6, 0.2) if focused else (0.15, 0.15, 0.18))


def _render_3d(state, data, buf, panel, cat_lut, group_opacity):
    cam = state.camera
    basis = cam.basis()
    if basis is None:
        return
    fwd, right, up = basis
    eye = cam.eye_position()
    fov_scale = math.tan(math.radians(cam.fov_deg) / 2.0)

    voxel_trace(
        data.material_map, data.skull_sdf, data.exterior_dist,
        cat_lut, group_opacity,
        buf,
        panel.x0, panel.y0, panel.w, panel.h,
        eye[0], eye[1], eye[2],
        fwd[0], fwd[1], fwd[2],
        right[0], right[1], right[2],
        up[0], up[1], up[2],
        fov_scale, state.N,
    )


def _render_crosshair_widget(state, buf, panel):
    cam = state.widget_camera()
    basis = cam.basis()
    if basis is None:
        return
    fwd, right, up = basis
    eye = cam.eye_position()
    fov_scale = math.tan(math.radians(cam.fov_deg) / 2.0)

    crosshair_3d(
        buf,
        panel.x0, panel.y0, panel.w, panel.h,
        eye[0], eye[1], eye[2],
        fwd[0], fwd[1], fwd[2],
        right[0], right[1], right[2],
        up[0], up[1], up[2],
        fov_scale, state.N,
        float(state.crosshair[0]), float(state.crosshair[1]),
        float(state.crosshair[2]),
    )


def _draw_gui(window, state, layers):
    """Draw sidebar with slice info, layer controls, and keybinds."""
    gui = window.get_gui()
    win_w, win_h = window.get_window_shape()

    if state.layout.sidebar_x0 >= win_w:
        return

    sb_x = state.layout.sidebar_x0 / win_w
    sb_w = 1.0 - sb_x
    with gui.sub_window("##sidebar", x=sb_x, y=0.0,
                         width=sb_w, height=1.0):
        if state.layout_mode == LayoutMode.SLICES:
            # Crosshair position
            i, j, k = state.crosshair
            gui.text(f"Crosshair")
            gui.text(f"  i={i}  j={j}  k={k}")
            gui.text("")

            # Slice indices per panel
            panel_names = ['Axial', 'Coronal', 'Sagittal']
            axis_letters = ['k', 'j', 'i']
            for pi in range(3):
                si = state.slice_index(pi)
                gui.text(f"{panel_names[pi]}  {axis_letters[pi]}={si}")
            gui.text("")

            # Layer controls
            gui.text("Layers")
            for idx, layer in enumerate(layers):
                layer.visible = gui.checkbox(f"{idx+1} {layer.name}", layer.visible)
                layer.opacity = gui.slider_float(f"##op{idx}", layer.opacity, 0.0, 1.0)

        if state.layout_mode == LayoutMode.THREE_D:
            gui.text("")
            gui.text("3D Groups")
            group_names = ['Brain', 'CSF', 'Dura', 'Skull', 'Choroid', 'Vessel']
            for gi, gname in enumerate(group_names):
                state.group_opacity[gi] = gui.slider_float(
                    f"{gname}##g{gi}", state.group_opacity[gi], 0.0, 1.0)

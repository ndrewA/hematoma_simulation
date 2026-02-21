"""Composite buffer operations: clear, crosshair, panel borders."""

import taichi as ti


@ti.kernel
def fill_rect(buf: ti.template(), x0: int, y0: int, w: int, h: int,
              r: float, g: float, b: float):
    """Fill a rectangular region with solid color."""
    color = ti.Vector([r, g, b])
    for px, py in ti.ndrange(w, h):
        buf[x0 + px, y0 + py] = color


@ti.kernel
def clear(buf: ti.template(), w: int, h: int, r: float, g: float, b: float):
    """Fill buffer region [0:w, 0:h] with solid color."""
    for px, py in ti.ndrange(w, h):
        buf[px, py] = ti.Vector([r, g, b])


@ti.kernel
def draw_crosshair(
    buf: ti.template(),
    px0: int, py0: int, pw: int, ph: int,
    cx: int, cy: int,
):
    """Draw dashed crosshair lines within a panel region."""
    color = ti.Vector([1.0, 1.0, 1.0])  # white
    alpha = 0.4
    dash = 4  # 4px on, 4px off
    # Horizontal line
    for x in range(pw):
        if (x // dash) % 2 == 0:
            bx = px0 + x
            by = py0 + cy
            if 0 <= by:
                old = buf[bx, by]
                buf[bx, by] = old * (1.0 - alpha) + color * alpha
    # Vertical line
    for y in range(ph):
        if (y // dash) % 2 == 0:
            bx = px0 + cx
            by = py0 + y
            old = buf[bx, by]
            buf[bx, by] = old * (1.0 - alpha) + color * alpha


@ti.kernel
def draw_panel_border(
    buf: ti.template(),
    px0: int, py0: int, pw: int, ph: int,
    r: float, g: float, b: float,
):
    """Draw 1px border around a panel."""
    color = ti.Vector([r, g, b])
    for x in range(pw):
        buf[px0 + x, py0] = color
        buf[px0 + x, py0 + ph - 1] = color
    for y in range(ph):
        buf[px0, py0 + y] = color
        buf[px0 + pw - 1, py0 + y] = color

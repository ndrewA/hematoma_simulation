"""Composite buffer operations: clear, crosshair, panel borders."""

import taichi as ti


@ti.kernel
def fill_rect(buf: ti.template(), x0: int, y0: int, w: int, h: int,
              r: float, g: float, b: float):
    """Fill a rectangular region with solid color."""
    color = ti.Vector([r, g, b])
    bw = buf.shape[0]
    bh = buf.shape[1]
    for px, py in ti.ndrange(w, h):
        bx = x0 + px
        by = y0 + py
        if 0 <= bx < bw and 0 <= by < bh:
            buf[bx, by] = color


@ti.kernel
def clear(buf: ti.template(), w: int, h: int, r: float, g: float, b: float):
    """Fill buffer region [0:w, 0:h] with solid color."""
    bw = buf.shape[0]
    bh = buf.shape[1]
    cw = ti.min(w, bw)
    ch = ti.min(h, bh)
    for px, py in ti.ndrange(cw, ch):
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
    bw = buf.shape[0]
    bh = buf.shape[1]
    # Horizontal line
    for x in range(pw):
        if (x // dash) % 2 == 0:
            bx = px0 + x
            by = py0 + cy
            if 0 <= bx < bw and 0 <= by < bh:
                old = buf[bx, by]
                buf[bx, by] = old * (1.0 - alpha) + color * alpha
    # Vertical line
    for y in range(ph):
        if (y // dash) % 2 == 0:
            bx = px0 + cx
            by = py0 + y
            if 0 <= bx < bw and 0 <= by < bh:
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
    bw = buf.shape[0]
    bh = buf.shape[1]
    if pw > 0 and ph > 0:
        for x in range(pw):
            bx = px0 + x
            by_lo = py0
            by_hi = py0 + ph - 1
            if 0 <= bx < bw:
                if 0 <= by_lo < bh:
                    buf[bx, by_lo] = color
                if 0 <= by_hi < bh:
                    buf[bx, by_hi] = color
        for y in range(ph):
            by = py0 + y
            bx_lo = px0
            bx_hi = px0 + pw - 1
            if 0 <= by < bh:
                if 0 <= bx_lo < bw:
                    buf[bx_lo, by] = color
                if 0 <= bx_hi < bw:
                    buf[bx_hi, by] = color

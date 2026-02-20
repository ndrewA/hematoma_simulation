"""Lightweight profiling: wall time + RSS memory tracking per step.

Usage:
    from preprocessing.profiling import step

    with step("load sources"):
        brain_mask, head_mask, affine = load_source_masks(raw)

    # Nesting works:
    with step("reconstruct falx"):
        with step("EDT pair"):
            ...

Output format:
    [load sources] 1.2s | RSS 1.4 GB (+0.3 GB) | peak 1.8 GB
      [EDT pair] 0.8s | RSS 1.1 GB (+0.1 GB) | peak 1.8 GB
"""

import resource
import threading
import time
from contextlib import contextmanager

_depth = threading.local()


def _get_depth():
    """Get current nesting depth (thread-local)."""
    return getattr(_depth, "value", 0)


def _set_depth(d):
    _depth.value = d


def _get_rss_mb():
    """Current RSS in MB via /proc/self/status (Linux) or getrusage fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (OSError, ValueError):
        pass
    # Fallback: ru_maxrss is peak, not current — better than nothing
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_peak_mb():
    """Peak RSS in MB (lifetime high-water mark)."""
    # Linux: ru_maxrss is in KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _fmt_mb(mb):
    """Format MB as human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def _fmt_delta(delta_mb):
    """Format delta with sign."""
    sign = "+" if delta_mb >= 0 else ""
    if abs(delta_mb) >= 1024:
        return f"{sign}{delta_mb / 1024:.2f} GB"
    return f"{sign}{delta_mb:.0f} MB"


@contextmanager
def step(name):
    """Context manager that tracks wall time and RSS memory for a named step.

    Prints a summary line on exit with elapsed time, current RSS,
    RSS delta, and peak RSS.  Nests cleanly with indentation.
    """
    depth = _get_depth()
    _set_depth(depth + 1)

    rss_start = _get_rss_mb()
    t_start = time.monotonic()

    try:
        yield
    finally:
        elapsed = time.monotonic() - t_start
        rss_end = _get_rss_mb()
        peak = _get_peak_mb()
        delta = rss_end - rss_start

        indent = "  " * depth
        print(f"{indent}[{name}] {elapsed:.1f}s"
              f" | RSS {_fmt_mb(rss_end)} ({_fmt_delta(delta)})"
              f" | peak {_fmt_mb(peak)}")

        _set_depth(depth)

"""Training-curve output: a CSV of the logged history and a JPG plot.

matplotlib is an optional dependency. If it is missing the CSV is still
written and the plot is skipped with a warning, so training never fails at
the finish line over a plotting library.
"""

import csv
import logging
from collections import deque

logger = logging.getLogger("dit")

# Chart surface and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e2e1dd"

# Categorical slots 1 and 2. Validated for CVD separation and contrast
# against the surface above.
COLOR_V_LOSS = "#2a78d6"
COLOR_X_MSE = "#eb6834"


def write_history_csv(history, path):
    """Write the (step, v_loss, x_mse) history so it can be re-plotted later."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "v_loss", "x_mse"])
        writer.writerows(history)


def _rolling_mean(values, window):
    """Trailing mean over ``window`` points, ramping up at the start."""
    buf, total, out = deque(), 0.0, []
    for v in values:
        buf.append(v)
        total += v
        if len(buf) > window:
            total -= buf.popleft()
        out.append(total / len(buf))
    return out


def _use_log_scale(values):
    """Log-scale the axis only when the curve spans a wide range."""
    finite = [v for v in values if v > 0]
    if len(finite) < 2:
        return False
    return max(finite) / min(finite) > 50


def _panel(ax, steps, values, color, label, window):
    ax.plot(steps, values, color=color, lw=1.0, alpha=0.35, label="logged")
    if window > 1:
        ax.plot(steps, _rolling_mean(values, window), color=color, lw=2.0,
                label=f"rolling mean ({window})")

    ax.set_ylabel(label, color=INK, fontsize=11)
    if _use_log_scale(values):
        ax.set_yscale("log")

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)


def save_loss_curve(history, path, log_every=None):
    """Plot v-loss and x-mse against training step and save to ``path``.

    Two stacked panels rather than one plot with two y-axes: the v-loss is
    the x-mse scaled by 1/(1-t)^2, so the two live on different scales and
    overlaying them on a shared axis would misrepresent both.
    """
    if len(history) < 2:
        logger.warning("not enough logged points to plot a curve")
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed on a training box
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping the loss curve plot")
        return False

    steps = [row[0] for row in history]
    v_loss = [row[1] for row in history]
    x_mse = [row[2] for row in history]
    window = max(1, len(history) // 50)

    fig, (ax_v, ax_x) = plt.subplots(
        2, 1, sharex=True, figsize=(8.0, 6.5), facecolor=SURFACE
    )

    _panel(ax_v, steps, v_loss, COLOR_V_LOSS, "v-loss", window)
    _panel(ax_x, steps, x_mse, COLOR_X_MSE, "x-mse", window)

    ax_x.set_xlabel("training step", color=INK_MUTED, fontsize=10)

    subtitle = "v-loss is the optimised objective; x-mse is the unweighted error"
    if log_every:
        subtitle += f"  ·  each point averages {log_every} batches"
    fig.suptitle("Training curves", color=INK, fontsize=13, x=0.125, y=0.98,
                 ha="left")
    fig.text(0.125, 0.925, subtitle, color=INK_MUTED, fontsize=9, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE, pil_kwargs={"quality": 92})
    plt.close(fig)
    return True

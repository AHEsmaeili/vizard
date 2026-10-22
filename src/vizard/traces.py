"""Training curves."""

import numpy as np
import matplotlib.pyplot as plt

from vizard.style import font_rc, modify_axis_spines


def plot_train_test_curves(
    ax,
    iterations,
    curves,
    *,
    y_label=None,
    x_label='Iteration',
    log_y=False,
    title=None,
    text=None,
    text_pos=(0.95, 0.6),
    text_fontsize=None,
    show_legend=True,
    fontsize=None,
    font_scale=1.0,
):
    """Loss curves over training iterations, one per entry of `curves`.

    Each curve is a dict with 'y' and 'label', plus any matplotlib line keywords
    (c, lw, alpha, ls). On a linear axis the y ticks sit at the rounded extremes of
    the first two curves; on a log axis at the axis limits.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        for curve in curves:
            kw = {k: v for k, v in curve.items() if k != 'y'}
            ax.plot(iterations, curve['y'], **kw)
        if y_label:
            ax.set_ylabel(y_label)
        if log_y:
            ax.set_yscale('log')
            ymin, ymax = ax.get_ylim()
            y_ticks = [ymin, ymax]
        else:
            first_two = np.column_stack([c['y'] for c in curves[:2]])
            y_ticks = [np.round(first_two.min(), 2), np.round(first_two.max(), 2)]

        if title:
            ax.set_title(title)
        ax.set_xlabel(x_label)

        if text:
            fs = text_fontsize if text_fontsize is not None else plt.rcParams['font.size']
            ax.text(
                text_pos[0],
                text_pos[1],
                text,
                transform=ax.transAxes,
                va='top',
                ha='right',
                fontsize=fs,
            )

        if show_legend:
            ax.legend(frameon=False)

        ax.patch.set_alpha(0)
        modify_axis_spines(
            ax,
            which=['x', 'y'],
            xticks_main=[int(iterations[0]), int(iterations[-1])],
            yticks_main=y_ticks,
        )
        return ax

"""Panel letters and figure-level font sizing."""

import matplotlib.pyplot as plt


def annotate_panels(axd, key_to_letter, *, offset=(-25, 15), letter_offsets=None, fontsize=11):
    """Bold panel letters at the top-left corner of each axis in `axd`, offset in
    points; `letter_offsets` overrides the offset for individual keys."""
    offsets = letter_offsets or {}
    for k, ltr in key_to_letter.items():
        ax = axd.get(k)
        if ax is None:
            continue
        off = offsets.get(k, offset)
        ax.annotate(
            ltr,
            xy=(0, 1),
            xycoords='axes fraction',
            xytext=off,
            textcoords='offset points',
            fontsize=fontsize,
            fontweight='bold',
            ha='right',
            va='bottom',
        )


def calibrated_fontsize(native_width_in, ref_fontsize=7, ref_width_in=6.7):
    """Font size that keeps the reference ratio of points to saved figure width.

    Figures sized this way look alike when placed at the same width in a
    document, whatever their own native width.
    """
    return ref_fontsize / ref_width_in * native_width_in


def two_pass_calibrated_fontsize(
    build_fn, nominal_width_in, pad_inches=0.0, ref_fontsize=7, ref_width_in=6.7
):
    """Calibrated font size measured on the figure as it will be saved.

    Saving with bbox_inches='tight' crops each figure by a different amount, so
    the nominal width is not the saved width. `build_fn(fontsize)` must build and
    return `(fig, ...)` without saving; it is drawn once at the nominal estimate,
    measured, closed, and the size is recalibrated to the measured width.
    `pad_inches` should match the eventual save call.
    """
    fontsize0 = calibrated_fontsize(nominal_width_in, ref_fontsize, ref_width_in)
    probe = build_fn(fontsize0)
    fig0 = probe[0] if isinstance(probe, tuple) else probe
    fig0.canvas.draw()
    actual_w = fig0.get_tightbbox(fig0.canvas.get_renderer()).width + 2 * pad_inches
    plt.close(fig0)
    return calibrated_fontsize(actual_w, ref_fontsize, ref_width_in)

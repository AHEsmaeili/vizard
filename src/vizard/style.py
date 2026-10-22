"""Fonts, tick formatting, spines, colorbars and figure saving shared by
every plotting module."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib_scalebar.scalebar import ScaleBar

base_fontsize = 7.0

# where save_figure writes relative filenames; set_figure_dir changes it
figure_dir = ''


def set_figure_dir(path):
    """Directory that save_figure writes relative filenames into; created if missing."""
    global figure_dir
    figure_dir = path
    os.makedirs(path, exist_ok=True)


def _fs(fontsize=None, font_scale=1.0, base=None):
    """Font size in points: `fontsize` if given, else `font_scale` times the base."""
    if base is None:
        base = base_fontsize
    if fontsize is not None:
        return float(fontsize)
    return float(base) * float(font_scale)


def font_rc(fontsize=None, font_scale=1.0):
    """rcParams for ``plt.rc_context`` that give every text element one font size."""
    efs = _fs(fontsize, font_scale)
    return {
        'font.size': efs,
        'axes.labelsize': efs,
        'axes.titlesize': efs,
        'xtick.labelsize': efs,
        'ytick.labelsize': efs,
        'legend.fontsize': efs,
        'legend.title_fontsize': efs,
        'figure.titlesize': efs,
    }


def set_mpl_colors(light_bg=True):
    """Black on white, or white on black with `light_bg=False`."""
    if light_bg:
        fc = 'white'
        ec = 'black'
    else:
        fc = 'black'
        ec = 'white'
    plt.rcParams.update(
        {
            "figure.facecolor": fc,
            "axes.facecolor": fc,
            "axes.edgecolor": ec,
            "axes.labelcolor": ec,
            "xtick.color": ec,
            "ytick.color": ec,
            "text.color": ec,
        }
    )


def save_figure(fig, filename, transparent=True, dpi=800, bbox_inches='tight', pad_inches=0):
    """Save with a transparent background, cropped tight to the drawn content."""
    fig.patch.set_alpha(0)
    fig.frameon = False

    for ax in fig.axes:
        ax.patch.set_alpha(0)

    save_path = os.path.join(figure_dir, filename)
    fig.savefig(
        save_path, transparent=transparent, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
    )


def modify_axis_spines(
    ax,
    which=None,
    base=1.0,
    xticks_main=None,
    yticks_main=None,
    xticks_twin=None,
    yticks_twin=None,
    yaxis_left=True,
    xaxis_bot=True,
    twinx=False,
    ax2=None,
    twiny=False,
):
    """Keep one spine per axis in `which`, trimmed to span exactly its ticks.

    Ticks default to the current ones; labels drop redundant decimals. A twin
    axis `ax2` gets the opposite spines when `twinx`/`twiny` is set.
    """
    if which is None:
        which = ['x', 'y']
    if isinstance(which, str):
        which = list(which)
    tick_locator = plticker.MultipleLocator(base=base)

    def handle_spines(ax_handle, axis_name, show_side, ticks):
        spines = ax_handle.spines
        if axis_name == 'x':
            spines['top'].set_visible(False)
            spines['bottom'].set_visible(False)
        elif axis_name == 'y':
            spines['left'].set_visible(False)
            spines['right'].set_visible(False)

        if axis_name == 'x':
            spines[show_side].set_visible(True)
            if ticks is None:
                ticks = ax_handle.get_xticks()
                ax_handle.xaxis.set_major_locator(tick_locator)
            ax_handle.set_xticks(ticks)
            ax_handle.set_xticklabels([_tick_fmt(t) for t in ticks])
            spines[show_side].set_bounds(min(ticks), max(ticks))

        elif axis_name == 'y':
            spines[show_side].set_visible(True)
            if ticks is None:
                ticks = ax_handle.get_yticks()
                ax_handle.yaxis.set_major_locator(tick_locator)
            ax_handle.set_yticks(ticks)
            ax_handle.set_yticklabels([_tick_fmt(t) for t in ticks])
            spines[show_side].set_bounds(min(ticks), max(ticks))

    if 'x' in which:
        handle_spines(ax, 'x', 'bottom' if xaxis_bot else 'top', xticks_main)
    if 'y' in which:
        handle_spines(ax, 'y', 'left' if yaxis_left else 'right', yticks_main)

    if ax2:
        if 'x' in which and twiny:
            handle_spines(ax2, 'x', 'top' if xaxis_bot else 'bottom', xticks_twin)
        else:
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)

        if 'y' in which and twinx:
            handle_spines(ax2, 'y', 'right' if yaxis_left else 'left', yticks_twin)
        else:
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)


def _tick_fmt(v):
    """Tick label with no redundant decimals, integers from 10 and 1e4 style from 10000."""
    if not np.isfinite(v):
        return str(v)
    if v != 0 and abs(v) >= 10000:
        exp = int(np.floor(np.log10(abs(v))))
        mant = v / (10**exp)
        if abs(mant - round(mant)) < 1e-9:
            return f'{int(round(mant))}e{exp}'
        return f'{mant:.1f}e{exp}'
    if v == int(v):
        return str(int(v))
    if abs(v) >= 10:
        return str(int(round(v)))
    for p in range(1, 7):
        r = round(v, p)
        if abs(r - v) < 1e-9 * max(abs(v), 1e-10):
            return f'{r:.{p}f}'
    return f'{v:.4f}'


def auto_ticks(lims, precision):
    """Two ticks at `lims` rounded to `precision` decimals, adding decimals until
    they stay distinct (-0.003 and 0.002 both round to 0.0 at two)."""
    lo, hi = float(lims[0]), float(lims[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo, lo + 1e-6]
    for p in range(precision, precision + 6):
        rlo, rhi = round(lo, p), round(hi, p)
        if rlo < rhi:
            return [rlo, rhi]
    return [lo, hi]


def latex_sci_fmt(v, _pos=None, decimals=1):
    """Tick label as mathtext m x 10^e, narrower than '1.0e-04'. Usable as a
    matplotlib FuncFormatter."""
    if v == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10**exp)
    mant_str = f'{mant:.{decimals}f}'
    if abs(round(float(mant_str)) - float(mant_str)) < 1e-9:
        mant_str = str(int(round(float(mant_str))))
    return rf'${mant_str}\times10^{{{exp}}}$'


def _nice_bounds(lo, hi, decimals=1):
    """Round `lo` down and `hi` up so the bounds still contain the data.

    `decimals` is a minimum: narrow spans get more places, so a 0.01-wide
    range is not padded out to (-0.1, 0.1).
    """
    span = max(hi - lo, 1e-12)
    decimals = max(decimals, int(np.ceil(-np.log10(span))) + 1)
    factor = 10**decimals
    return float(np.floor(lo * factor) / factor), float(np.ceil(hi * factor) / factor)


def _auto_scale_exponent(values, threshold=1e-3):
    """Power of ten to factor out of an axis whose largest value is below
    `threshold`, so its ticks do not read 0.0000. Zero otherwise; values of
    order 1e-2 read fine as plain decimals."""
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite) & (finite != 0)]
    if finite.size == 0:
        return 0
    max_abs = np.max(np.abs(finite))
    return int(np.floor(np.log10(max_abs))) if 0 < max_abs < threshold else 0


def _auto_labelpad(ax, axis='x', extra=2.0, fallback=None):
    """Labelpad in points that clears the tick labels, measured on a draw of
    the figure, or estimated from the font size without a renderer. Call it
    after the ticks are set."""
    fb = fallback if fallback is not None else plt.rcParams['axes.labelpad']
    fig = ax.figure
    tls = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()
    tls = [t for t in tls if t.get_text()]
    if not tls:
        return fb

    try:
        fig.canvas.draw()
        rdr = fig.canvas.get_renderer()
        if axis == 'x':
            sz = max(t.get_window_extent(rdr).height for t in tls)
        else:
            sz = max(t.get_window_extent(rdr).width for t in tls)
        return sz * 72.0 / fig.dpi + extra
    except Exception:
        fs = plt.rcParams['font.size']
        if axis == 'x':
            return fs + extra
        max_chars = max((len(t.get_text()) for t in tls), default=1)
        return max_chars * fs * 0.55 + extra


def _resolve_labelpad(ax, labelpad, axis='x', extra=2.0):
    """`labelpad` when given, else _auto_labelpad."""
    return labelpad if labelpad is not None else _auto_labelpad(ax, axis=axis, extra=extra)


def add_styled_colorbar(
    im,
    ax=None,
    label=None,
    ticks=None,
    cax=None,
    orientation='vertical',
    vrotation=90,
    fraction=0.046,
    pad=0.04,
    labelpad=None,
    label_pos=None,
    fontsize=None,
    font_scale=1.0,
):
    """Colorbar without outline or tick marks, ticked at the colour limits by default."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ticks is None:
            ticks = im.get_clim()

        if cax is None:
            cbar = plt.colorbar(
                im, ax=ax, fraction=fraction, pad=pad, ticks=ticks, orientation=orientation
            )
        else:
            cbar = plt.colorbar(im, cax=cax, ticks=ticks, orientation=orientation)

        if label:
            lp = labelpad if labelpad is not None else 0
            if orientation == 'vertical':
                cbar.set_label(label, rotation=vrotation, labelpad=lp)
                if label_pos is not None:
                    cbar.ax.yaxis.set_label_coords(*label_pos)
            else:
                cbar.set_label(label, labelpad=lp)
                if label_pos is not None:
                    cbar.ax.xaxis.set_label_coords(*label_pos)

        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
        return cbar


def add_custom_scalebar(
    ax, scale_unit, label_fmt, loc_parent, loc_scale, pos=None, known_obj=None, **kwargs
):
    """Frameless matplotlib-scalebar bar; `pos` places it in axes coordinates."""
    sb_params = {
        'dimension': 'si-length',
        'location': loc_parent,
        'box_alpha': 0.0,
        'scale_loc': loc_scale,
        'border_pad': 0.5,
        'sep': 5,
        **kwargs,
    }
    if pos:
        sb_params.update({'bbox_to_anchor': pos, 'bbox_transform': ax.transAxes})
    sb = ScaleBar(scale_unit, 'm', **sb_params)
    if known_obj:
        sb.size_of_known_object = known_obj
    sb.label_formatter = label_fmt
    ax.add_artist(sb)


def plot_image_schema(ax, image_path=None, hide_axes=True, **kwargs):
    """Draw an image file (a schematic, say) into `ax`; a missing file leaves
    the axis blank as a placeholder."""
    if image_path and os.path.exists(image_path):
        import matplotlib.image as mpimg

        img = mpimg.imread(image_path)
        ax.imshow(img, **kwargs)
    if hide_axes:
        ax.axis('off')
    return ax


def clean_colorbar(
    cbar, vmin, vmax, label=None, frameon=False, tick_length=0, labelpad=10, vrotation=90
):
    """Two ticks at the ends, labelled with as few decimals as tell them apart."""
    cbar.outline.set_visible(frameon)
    cbar.ax.tick_params(length=tick_length, which='major')
    cbar.ax.tick_params(length=0, which='minor')
    cbar.ax.minorticks_off()

    def _auto_fmt(lo, hi):
        for dp in range(0, 5):
            sl = f'{lo:.{dp}f}'
            sh = f'{hi:.{dp}f}'
            if sl != sh:
                return sl, sh
        return f'{lo:.4g}', f'{hi:.4g}'

    labels = list(_auto_fmt(vmin, vmax))

    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(labels)
    if label:
        cbar.set_label(label, rotation=vrotation, labelpad=labelpad)


def compact_count_label(v):
    """Tick label for counts: '0' below one, '5e3' style from a thousand up."""
    if abs(v) < 1:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    if exp >= 3:
        return f"{v / 10**exp:.0f}e{exp}"
    return f"{v:.0f}"

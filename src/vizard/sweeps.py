"""Simulation sweeps: response curves over one swept parameter, and feature
maps over a grid of two."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

from vizard.style import font_rc, auto_ticks, _tick_fmt, modify_axis_spines, add_styled_colorbar
from vizard.matrices import matrix_cbar_inset


def plot_sweep_curves(
    features,
    feature_names,
    sweep_vals,
    subj_idx,
    color_vals,
    *,
    sel_features=None,
    feature_labels=None,
    color_label=None,
    cmap='inferno',
    axes=None,
    panel_size=(2.0, 2.0),
    wspace=0.5,
    s=8,
    scatter_alpha=0.35,
    scatter_edgecolor='none',
    scatter_edgewidth=0.5,
    show_line=True,
    line_lw=1.2,
    line_alpha=0.75,
    line_color=None,
    line_edge_color=None,
    line_edge_width=None,
    fit_p0=None,
    fit_bounds=(-np.inf, np.inf),
    show_legend=True,
    legend_label_prefix='',
    legend_title=None,
    legend_reverse=False,
    legend_ax_idx=-1,
    legend_kw=None,
    show_cbar=False,
    x_label=None,
    x_ticks=None,
    x_tick_fmt=None,
    labelpad=-10,
    fontsize=None,
    font_scale=1.0,
):
    """Response of several features to one swept parameter, one curve per unit
    (subject, connectome, ...) coloured by a per-unit value.

    Rows of `features` are simulations; `subj_idx` says which unit each row
    belongs to and `sweep_vals` gives the swept values in row order within a
    unit. Each unit's points are drawn with an ``a*exp(-b*x)+c`` fit through
    them (`fit_p0`/`fit_bounds` seed and bound the fit; it falls back to the
    raw points if it fails). `line_edge_color` outlines the lines so they stay
    visible over the scatter. The legend labels units by their colour value,
    prefixed by `legend_label_prefix` unless a `legend_title` is drawn beside it.
    """
    feat_arr = np.asarray(features)
    feature_names = list(feature_names)
    sel_features = list(sel_features) if sel_features is not None else feature_names
    feature_labels = feature_labels or {}
    color_arr = np.asarray(color_vals, dtype=float)
    legend_kw = legend_kw or {}

    n_panels = len(sel_features)
    n_subj = len(color_arr)
    leg_panel = legend_ax_idx % n_panels

    norm = mcolors.Normalize(vmin=color_arr.min(), vmax=color_arr.max())
    cmap_obj = plt.get_cmap(cmap)

    with plt.rc_context(font_rc(fontsize, font_scale)):
        if axes is not None:
            axs = list(axes)
            fig = axs[0].figure
        else:
            fig, axs = plt.subplots(
                1,
                n_panels,
                figsize=(panel_size[0] * n_panels, panel_size[1]),
                gridspec_kw={'wspace': wspace},
            )
            fig.patch.set_alpha(0)
            if n_panels == 1:
                axs = [axs]

        legend_handles = []

        for ax_idx, (ax, feat_name) in enumerate(zip(axs, sel_features)):
            feat_col = feature_names.index(feat_name)
            ax.patch.set_alpha(0)

            y_all = []
            for si in range(n_subj):
                mask = subj_idx == si
                y_vals = feat_arr[mask, feat_col]
                col = cmap_obj(norm(color_arr[si]))
                lc = col if line_color is None else line_color

                ax.scatter(
                    sweep_vals,
                    y_vals,
                    color=col,
                    alpha=scatter_alpha,
                    s=s,
                    zorder=4,
                    edgecolors=scatter_edgecolor,
                    linewidths=scatter_edgewidth,
                )

                if show_line and len(y_vals) >= 3:

                    def _exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c

                    try:
                        _y0, _y1 = float(y_vals[0]), float(y_vals[-1])
                        _p0 = (
                            fit_p0
                            if fit_p0 is not None
                            else [
                                _y0 - _y1,
                                1.0 / max(float(sweep_vals[-1] - sweep_vals[0]), 1e-10),
                                _y1,
                            ]
                        )
                        _popt, _ = curve_fit(
                            _exp_decay, sweep_vals, y_vals, p0=_p0, bounds=fit_bounds, maxfev=8000
                        )
                        y_smooth = _exp_decay(sweep_vals, *_popt)
                    except Exception:
                        y_smooth = y_vals
                    _pe = None
                    if line_edge_color is not None:
                        _ew = line_edge_width if line_edge_width is not None else line_lw + 2
                        _pe = [mpe.withStroke(linewidth=_ew, foreground=line_edge_color)]
                    ax.plot(
                        sweep_vals,
                        y_smooth,
                        color=lc,
                        lw=line_lw,
                        alpha=line_alpha,
                        zorder=100,
                        path_effects=_pe,
                    )

                if ax_idx == 0:
                    _lbl = (
                        f'{legend_label_prefix}: {color_arr[si]:.2f}'
                        if legend_label_prefix and legend_title is None
                        else f'{color_arr[si]:.2f}'
                    )
                    legend_handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            color=col,
                            lw=line_lw if show_line else 0,
                            marker='o',
                            markersize=4,
                            label=_lbl,
                        )
                    )
                y_all.append(y_vals)

            y_concat = np.concatenate(y_all)
            _xt = (
                list(x_ticks)
                if x_ticks is not None
                else auto_ticks([float(sweep_vals[0]), float(sweep_vals[-1])], 2)
            )
            _yt = auto_ticks([float(np.nanmin(y_concat)), float(np.nanmax(y_concat))], 2)
            modify_axis_spines(ax, which=['x', 'y'], xticks_main=_xt, yticks_main=_yt)
            if x_tick_fmt is not None:
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(x_tick_fmt))
            ax.set_xlabel(x_label, labelpad=labelpad)
            ax.set_ylabel(feature_labels.get(feat_name, feat_name), labelpad=labelpad)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        if show_legend and legend_handles:
            _fs = (fontsize or 8) * font_scale
            _lkw = dict(
                frameon=False, fontsize=_fs, handlelength=1.2, borderpad=0, labelspacing=0.25
            )
            _lkw.update(legend_kw)
            _handles = legend_handles[::-1] if legend_reverse else legend_handles
            _leg = axs[leg_panel].legend(handles=_handles, **_lkw)

            if legend_title is not None:
                _fig = axs[leg_panel].figure
                _fig.canvas.draw()
                _bb = _leg.get_window_extent().transformed(axs[leg_panel].transAxes.inverted())
                axs[leg_panel].text(
                    _bb.x1 + 0.04,
                    (_bb.y0 + _bb.y1) / 2,
                    legend_title,
                    transform=axs[leg_panel].transAxes,
                    rotation=90,
                    ha='left',
                    va='center',
                    fontsize=_fs,
                )

        if show_cbar:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            _cax = axs[-1].inset_axes([1.05, 0.0, 0.06, 1.0])
            add_styled_colorbar(
                sm,
                cax=_cax,
                label=color_label or '',
                ticks=[float(color_arr.min()), float(color_arr.max())],
                fontsize=fontsize,
                font_scale=font_scale,
            )

    return fig, axs


def plot_sweep_heatmaps(
    features,
    feature_names,
    row_vals,
    col_vals,
    *,
    sel_features=None,
    feature_labels=None,
    cmap='inferno',
    axes=None,
    panel_size=(2.2, 2.0),
    wspace=0.5,
    show_cbar=True,
    cbar_inset=None,
    transpose=False,
    x_label=None,
    y_label=None,
    x_ticks=None,
    x_tick_fmt=None,
    y_ticks=None,
    y_tick_fmt=None,
    ytick_margin=0.1,
    labelpad=-10,
    fontsize=None,
    font_scale=1.0,
):
    """Heatmaps of several features over a two-parameter grid.

    Rows of `features` follow the grid in ``np.meshgrid(row_vals, col_vals,
    indexing='ij')`` order. By default `col_vals` runs along x and `row_vals`
    along y; `transpose` swaps them. Tick arguments always refer to the axis
    they are drawn on. Each panel gets a two-tick colorbar.
    """
    feat_arr = np.asarray(features)
    feature_names = list(feature_names)
    sel_features = list(sel_features) if sel_features is not None else feature_names
    feature_labels = feature_labels or {}

    n_rows = len(row_vals)
    n_cols = len(col_vals)
    n_panels = len(sel_features)

    feat_grid = feat_arr.reshape(n_rows, n_cols, len(feature_names))

    _cbar_inset = list(cbar_inset) if cbar_inset is not None else list(matrix_cbar_inset)

    with plt.rc_context(font_rc(fontsize, font_scale)):
        if axes is not None:
            axs = list(axes)
            fig = axs[0].figure
        else:
            fig, axs = plt.subplots(
                1,
                n_panels,
                figsize=(panel_size[0] * n_panels, panel_size[1]),
                gridspec_kw={'wspace': wspace},
            )
            fig.patch.set_alpha(0)
            if n_panels == 1:
                axs = [axs]

        # transposing only changes which values are plotted where; tick
        # arguments already refer to the physical axis
        xv, yv = (row_vals, col_vals) if transpose else (col_vals, row_vals)
        _user_xticks, _user_xfmt = x_ticks, x_tick_fmt
        _user_yticks, _user_yfmt = y_ticks, y_tick_fmt

        for ax, feat_name in zip(axs, sel_features):
            feat_col = feature_names.index(feat_name)
            z = feat_grid[:, :, feat_col]
            z_plot = z.T if transpose else z
            ax.patch.set_alpha(0)

            pcm = ax.pcolormesh(xv, yv, z_plot, cmap=cmap, shading='auto')

            _xt = list(_user_xticks) if _user_xticks is not None else [float(xv[0]), float(xv[-1])]
            _yt = list(_user_yticks) if _user_yticks is not None else [float(yv[0]), float(yv[-1])]
            ax.set_xticks(_xt)
            if callable(_user_xfmt):
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_user_xfmt))
            elif _user_xfmt is not None:
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(_user_xfmt))
            else:
                ax.set_xticklabels([_tick_fmt(t) for t in _xt])
            ax.set_yticks(_yt)
            if callable(_user_yfmt):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(_user_yfmt))
            elif _user_yfmt is not None:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(_user_yfmt))
            else:
                ax.set_yticklabels([_tick_fmt(t) for t in _yt])

            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)

            # margins keep the end labels clear of neighbouring panels
            _x_range = float(xv[-1] - xv[0])
            _y_range = float(yv[-1] - yv[0])
            ax.set_xlim(
                float(xv[0]) - _x_range * ytick_margin, float(xv[-1]) + _x_range * ytick_margin
            )
            ax.set_ylim(
                float(yv[0]) - _y_range * ytick_margin, float(yv[-1]) + _y_range * ytick_margin
            )

            # square box, sized the way imshow(aspect='equal') sizes matrix panels
            ax.set_aspect(_x_range / _y_range, adjustable='box')

            ax.set_xlabel(x_label, labelpad=labelpad)
            ax.set_ylabel(y_label, labelpad=labelpad)

            if show_cbar:
                _cax = ax.inset_axes(_cbar_inset)
                add_styled_colorbar(
                    pcm,
                    cax=_cax,
                    label=feature_labels.get(feat_name, feat_name),
                    ticks=[float(z.min()), float(z.max())],
                    fontsize=fontsize,
                    font_scale=font_scale,
                )

    return fig, axs

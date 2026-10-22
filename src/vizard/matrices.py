"""Connectivity matrices and BOLD traces."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from vizard.style import (
    font_rc,
    _resolve_labelpad,
    add_custom_scalebar,
    add_styled_colorbar,
    clean_colorbar,
    modify_axis_spines,
    save_figure,
    set_mpl_colors,
)

# shared by all matrix panels so their colorbars and titles line up
matrix_cbar_inset = (1.02, 0.0, 0.04, 1.0)
matrix_cbar_labelpad = -10
matrix_title_pad = 2
matrix_title_kw = dict(fontweight='normal', loc='center')


def plot_sc_template(
    sc_matrix,
    ax=None,
    figsize=None,
    cmap='magma',
    vmin=0.0,
    vmax=None,
    vmax_percentile=99,
    show_mask=True,
    mask_alpha=0.55,
    mask_color='white',
    show_interhemi_boxes=True,
    interhemi_edgecolor='lime',
    interhemi_lw=1.2,
    node_limit=None,
    overlay_limit=None,
    show_cbar=True,
    cbar_label='Value',
    cbar_labelpad=matrix_cbar_labelpad,
    cbar_inset=matrix_cbar_inset,
    cbar_tick_decimals=2,
    title=None,
    x_label='Node',
    y_label='Node',
    fontsize=None,
    font_scale=1.0,
):
    """Structural connectivity matrix with the interhemispheric quadrants
    outlined and the intrahemispheric ones veiled.

    The quadrants assume two contiguous, equal hemisphere blocks.
    `overlay_limit` confines them to the first nodes where that holds (the
    cortical block, when subcortical nodes interleave left and right) while
    every node is still drawn; `node_limit` crops the matrix itself.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.0, 3.0))

        sc = np.asarray(sc_matrix)
        if node_limit is not None:
            sc = sc[:node_limit, :node_limit]
        n = sc.shape[0]
        half = (overlay_limit if overlay_limit is not None else n) / 2.0
        v_top = vmax if vmax is not None else float(np.percentile(sc, vmax_percentile))

        im = ax.imshow(
            sc,
            cmap=cmap,
            origin='upper',
            aspect='equal',
            vmin=vmin,
            vmax=v_top,
            interpolation='nearest',
        )

        if show_mask:
            rgb = mcolors.to_rgb(mask_color)
            face = (*rgb, mask_alpha)
            ax.add_patch(mpatches.Rectangle((-0.5, -0.5), half, half, linewidth=0, facecolor=face))
            ax.add_patch(
                mpatches.Rectangle(
                    (half - 0.5, half - 0.5), half, half, linewidth=0, facecolor=face
                )
            )

        if show_interhemi_boxes:
            box_kw = dict(linewidth=interhemi_lw, edgecolor=interhemi_edgecolor, facecolor='none')
            ax.add_patch(mpatches.Rectangle((half - 0.5, -0.5), half, half, **box_kw))
            ax.add_patch(mpatches.Rectangle((-0.5, half - 0.5), half, half, **box_kw))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title, pad=matrix_title_pad, **matrix_title_kw)
        ax.patch.set_alpha(0)

        if show_cbar:
            cax = ax.inset_axes(cbar_inset)
            fmt = f'%.{cbar_tick_decimals}f'
            add_styled_colorbar(
                im,
                cax=cax,
                label=cbar_label,
                ticks=[vmin, v_top],
                labelpad=cbar_labelpad,
                fontsize=fontsize,
                font_scale=font_scale,
            )
            cax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter(fmt))
        return ax


def plot_fc_template(
    fc_matrix,
    ax=None,
    figsize=None,
    cmap='inferno',
    vmin=None,
    vmax=None,
    show_homotopic=True,
    homotopic_color='lime',
    homotopic_lw=1.5,
    homotopic_ls='-',
    homotopic_alpha=1.0,
    show_cbar=True,
    cbar_label='Value',
    cbar_labelpad=matrix_cbar_labelpad,
    cbar_inset=matrix_cbar_inset,
    cbar_tick_decimals=1,
    title=None,
    x_label='Node',
    y_label='Node',
    labelpad=None,
    fontsize=None,
    font_scale=1.0,
):
    """Functional connectivity matrix with the homotopic diagonal marked in the
    upper triangle, for nodes ordered left hemisphere then right."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.0, 3.0))

        fc = np.asarray(fc_matrix)
        n = fc.shape[0]
        vmin = float(np.nanmin(fc)) if vmin is None else vmin
        vmax = float(np.nanmax(fc)) if vmax is None else vmax
        im = ax.imshow(
            fc,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='upper',
            aspect='equal',
            interpolation='nearest',
        )

        if show_homotopic:
            half = n // 2
            ax.plot(
                [half, n - 1],
                [0, half - 1],
                color=homotopic_color,
                lw=homotopic_lw,
                ls=homotopic_ls,
                alpha=homotopic_alpha,
                solid_capstyle='butt',
                zorder=3,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if x_label:
            ax.set_xlabel(x_label, labelpad=_resolve_labelpad(ax, labelpad, 'x'))
        if y_label:
            ax.set_ylabel(y_label, labelpad=_resolve_labelpad(ax, labelpad, 'y'))
        if title:
            ax.set_title(title, pad=matrix_title_pad, **matrix_title_kw)
        ax.patch.set_alpha(0)

        if show_cbar:
            cax = ax.inset_axes(cbar_inset)
            fmt = f'%.{cbar_tick_decimals}f'
            add_styled_colorbar(
                im,
                cax=cax,
                label=cbar_label,
                ticks=[vmin, vmax],
                labelpad=cbar_labelpad,
                fontsize=fontsize,
                font_scale=font_scale,
            )
            cax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter(fmt))
        return ax


def plot_fcd_template(
    fcd_matrix,
    ax=None,
    figsize=None,
    cmap='viridis',
    vmin=0.0,
    vmax=1.0,
    show_cbar=True,
    cbar_label='Value',
    cbar_labelpad=matrix_cbar_labelpad,
    cbar_inset=matrix_cbar_inset,
    cbar_tick_decimals=1,
    title=None,
    x_label='Window',
    y_label='Window',
    labelpad=None,
    fontsize=None,
    font_scale=1.0,
):
    """Functional connectivity dynamics matrix, window against window."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.0, 3.0))

        m = np.asarray(fcd_matrix)
        im = ax.imshow(
            m,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='upper',
            aspect='equal',
            interpolation='nearest',
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if x_label:
            ax.set_xlabel(x_label, labelpad=_resolve_labelpad(ax, labelpad, 'x'))
        if y_label:
            ax.set_ylabel(y_label, labelpad=_resolve_labelpad(ax, labelpad, 'y'))
        if title:
            ax.set_title(title, pad=matrix_title_pad, **matrix_title_kw)
        ax.patch.set_alpha(0)

        if show_cbar:
            cax = ax.inset_axes(cbar_inset)
            fmt = f'%.{cbar_tick_decimals}f'
            add_styled_colorbar(
                im,
                cax=cax,
                label=cbar_label,
                ticks=[vmin, vmax],
                labelpad=cbar_labelpad,
                fontsize=fontsize,
                font_scale=font_scale,
            )
            cax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter(fmt))
        return ax


def plot_sample_bold(
    data,
    fig=None,
    ax=None,
    tstart=0,
    tend=None,
    nstep=1,
    lw=0.5,
    alpha=0.4,
    color='slategray',
    labelpad=5,
    scalebar=False,
    tr=1.0,
    pos_xbar=None,
    title=None,
    title_fontsize=None,
    savefig=False,
    fsave_suffix='synthetic',
    fontsize=None,
    font_scale=1.0,
    stack=False,
    stack_n=None,
    stack_offset=None,
):
    """BOLD time series (time x region), overlaid in one colour or, with
    `stack`, each region on its own baseline `stack_offset` apart (three
    standard deviations by default) for the first `stack_n` regions.
    `scalebar` swaps the axes for a time scale bar."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        set_mpl_colors(True)
        if fig is None:
            fig, ax = plt.subplots()
        time_end = data.shape[0] if tend is None else tend
        x = np.linspace(tstart, time_end, data.shape[0])
        plot_data = data[:, ::nstep]
        if stack:
            if stack_n is not None:
                plot_data = plot_data[:, :stack_n]
            offset = stack_offset if stack_offset is not None else np.nanstd(plot_data) * 3.0
            offsets = np.arange(plot_data.shape[1]) * offset
            ax.plot(x, plot_data + offsets[None, :], color=color, alpha=alpha, lw=lw)
        else:
            ax.plot(x, plot_data, color=color, alpha=alpha, lw=lw)
        ax.patch.set_alpha(0)

        if scalebar:
            modify_axis_spines(ax, which=[], yticks_main=None, xticks_main=None)
            ax.set_xlabel('')
            ax.set_ylabel('')
            add_custom_scalebar(
                ax,
                tr,
                lambda x, u: f'{x:g} s',
                'lower right',
                'bottom',
                pos=pos_xbar,
                fixed_value=50,
            )
            ax.axis('off')
        elif stack:
            modify_axis_spines(ax, which=['x'], yticks_main=None, xticks_main=[tstart, time_end])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time (frame)', labelpad=labelpad)
            ax.set_ylabel('Region (stacked)', labelpad=labelpad)
        else:
            modify_axis_spines(
                ax, which=['x', 'y'], yticks_main=[0, data.max()], xticks_main=[tstart, time_end]
            )
            ax.set_xlabel('Time (frame)', labelpad=labelpad)
            ax.set_ylabel('BOLD', labelpad=labelpad)

        if title:
            fs = title_fontsize or plt.rcParams.get('axes.titlesize', 8)
            ax.set_title(title, fontsize=fs, fontweight='normal', pad=2)
        if savefig:
            set_mpl_colors(True)
            save_figure(fig, f'bold_sample_{fsave_suffix}.svg')


def plot_distance_matrix(
    dist,
    *,
    ax=None,
    figsize=None,
    cmap='viridis',
    show_cbar=True,
    cbar_label=None,
    cbar_labelpad=10,
    title=None,
    x_label=None,
    y_label=None,
    show_ticks=True,
    show_title=True,
    show_margin=True,
    fontsize=None,
    font_scale=1.0,
):
    """Distances between matched items of two sets, rows against columns.

    The default title reports how often a row is nearest its own column
    (accuracy) and, with `show_margin`, the mean off-diagonal minus mean
    diagonal distance -- still informative once accuracy saturates at 100%.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.0, 3.0))
        ns = dist.shape[0]
        acc = float(np.mean(np.argmin(dist, axis=1) == np.arange(ns)))
        diag_mean = float(np.mean(np.diag(dist)))
        offdiag_mean = float(np.mean(dist[~np.eye(ns, dtype=bool)]))
        margin = offdiag_mean - diag_mean

        im = ax.imshow(dist, cmap=cmap, origin='upper', aspect='equal')
        if show_title:
            title_str = title or f'Acc: {acc * 100:.1f}%'
            if show_margin and title is None:
                title_str += f'\nMargin: {margin:.2f}'
            ax.set_title(title_str)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if show_ticks:
            # 1-indexed, so 50 items read as 1..50 rather than 0..49
            ax.set_xticks([0, ns - 1])
            ax.set_yticks([0, ns - 1])
            ax.set_xticklabels([1, ns])
            ax.set_yticklabels([1, ns])
            ax.tick_params(length=0)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.patch.set_alpha(0)

        if show_cbar:
            cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
            cb = plt.colorbar(im, cax=cax)
            clean_colorbar(
                cb, float(dist.min()), float(dist.max()), label=cbar_label, labelpad=cbar_labelpad
            )
        return ax

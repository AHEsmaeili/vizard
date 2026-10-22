"""Scatter plots with linear fits, alone, per group, in grids and binned."""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats

try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
from vizard.style import (
    auto_ticks,
    font_rc,
    clean_colorbar,
    compact_count_label,
    modify_axis_spines,
    save_figure,
)


def plot_regression_layer(
    ax, x, y, color, reg_color, s, alpha, reg_lw, label=None, show_text=True, text_box_alpha=0.5
):
    """Scatter, dashed least-squares line and, with `show_text`, the Pearson and
    Spearman r in the top-left corner."""
    x = x.values if hasattr(x, 'values') else np.array(x)
    y = y.values if hasattr(y, 'values') else np.array(y)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return

    ax.scatter(x, y, color=color, s=s, alpha=alpha, label=label)

    res = stats.linregress(x, y)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_plot = res.intercept + res.slope * x_plot
    ax.plot(x_plot, y_plot, color=reg_color, linestyle='--', linewidth=reg_lw, alpha=0.8)

    if show_text:
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        corr_text = rf"$r_{{P}}: {r_p:.2f}$" "\n" rf"$r_{{S}}: {r_s:.2f}$"
        ax.text(
            0.05,
            0.95,
            corr_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=text_box_alpha, linewidth=0),
        )


def plot_scatter_with_regression(
    ax,
    x,
    y,
    *,
    c=None,
    color='slategray',
    cmap=None,
    s=6,
    alpha=0.7,
    edgecolors='none',
    plot_regression=True,
    reg_color='indianred',
    reg_lw=2.0,
    reg_ls='--',
    reg_alpha=0.85,
    show_corr_text=True,
    corr_loc=(0.05, 0.97),
    corr_kind='pearson_spearman',
    corr_decimals=2,
    text_ha='left',
    text_va='top',
    text_box_alpha=0.5,
    text_fontweight='normal',
    x_label=None,
    y_label=None,
    labelpad=2,
    x_ticks=None,
    y_ticks=None,
    x_round=1,
    y_round=2,
    x_lim=None,
    y_lim=None,
    modify_spines=True,
    spine_which=('x', 'y'),
    subsample=None,
    rng_seed=None,
    fontsize=None,
    font_scale=1.0,
):
    """Scatter of y against x with a least-squares line and correlation text.

    `c` colours points by a value through `cmap` in place of `color`;
    `subsample` draws a fraction (float) or count (int) of the points. Ticks
    sit at the data range unless given.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        x_arr = np.asarray(x.values if hasattr(x, 'values') else x).ravel()
        y_arr = np.asarray(y.values if hasattr(y, 'values') else y).ravel()

        if c is not None:
            c_arr = np.asarray(c.values if hasattr(c, 'values') else c).ravel()
            mask = (
                ~np.isnan(x_arr) & ~np.isnan(y_arr) & ~np.isnan(c_arr.astype(float, copy=False))
                if np.issubdtype(c_arr.dtype, np.number)
                else (~np.isnan(x_arr) & ~np.isnan(y_arr))
            )
        else:
            c_arr = None
            mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)

        xm, ym = x_arr[mask], y_arr[mask]
        cm = c_arr[mask] if c_arr is not None else None

        if subsample is not None and len(xm) > 0:
            n_keep = int(subsample * len(xm)) if isinstance(subsample, float) else int(subsample)
            n_keep = max(min(n_keep, len(xm)), 1)
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(len(xm), n_keep, replace=False)
            xm, ym = xm[idx], ym[idx]
            if cm is not None:
                cm = cm[idx]

        scatter_color = cm if cm is not None else color
        ax.scatter(
            xm,
            ym,
            c=scatter_color,
            cmap=cmap if cm is not None else None,
            s=s,
            alpha=alpha,
            edgecolors=edgecolors,
        )

        if plot_regression and len(xm) > 2:
            res = stats.linregress(xm, ym)
            xp = np.linspace(xm.min(), xm.max(), 100)
            ax.plot(
                xp,
                res.intercept + res.slope * xp,
                color=reg_color,
                ls=reg_ls,
                lw=reg_lw,
                alpha=reg_alpha,
            )

        if show_corr_text and corr_kind and len(xm) > 2:
            r_p = stats.pearsonr(xm, ym)[0]
            r_s = stats.spearmanr(xm, ym)[0]
            if corr_kind == 'pearson_spearman':
                txt = rf"$r_{{P}}: {r_p:.{corr_decimals}f}$ " rf"$r_{{S}}: {r_s:.{corr_decimals}f}$"
            elif corr_kind == 'pearson':
                txt = rf"$r: {r_p:.{corr_decimals}f}$"
            elif corr_kind == 'spearman':
                txt = rf"$\rho: {r_s:.{corr_decimals}f}$"
            else:
                txt = ''
            if txt:
                ax.text(
                    corr_loc[0],
                    corr_loc[1],
                    txt,
                    transform=ax.transAxes,
                    ha=text_ha,
                    va=text_va,
                    fontweight=text_fontweight,
                    bbox=dict(
                        boxstyle='round', facecolor='white', alpha=text_box_alpha, linewidth=0
                    ),
                )

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_label is not None:
            ax.set_xlabel(x_label, labelpad=labelpad)
        if y_label is not None:
            ax.set_ylabel(y_label, labelpad=labelpad)
        ax.patch.set_alpha(0)

        if modify_spines and len(xm) > 1:
            xt = x_ticks if x_ticks is not None else auto_ticks([xm.min(), xm.max()], x_round)
            yt = y_ticks if y_ticks is not None else auto_ticks([ym.min(), ym.max()], y_round)
            modify_axis_spines(ax, which=list(spine_which), xticks_main=xt, yticks_main=yt)
        return ax


def plot_agreement_hexbin(
    x,
    y,
    ax,
    *,
    x_fit=None,
    y_fit=None,
    lower_limit=None,
    cmap='inferno',
    gridsize=50,
    axis_labelpad=-5,
    show_cbar=True,
    cbar_label='Count (log)',
    cbar_labelpad=10,
    compact_cbar=False,
    show_text=True,
    show_eq=True,
    text_loc=(0.05, 0.95),
    text_fontweight='bold',
    reg_color='k',
    reg_lw=2.0,
    x_label=None,
    y_label=None,
    title=None,
    x_round=2,
    y_round=2,
):
    """Log-count hexbin of observed (x) against predicted (y) values, with a
    least-squares line and its Spearman r.

    The fit and the correlation use `x_fit`/`y_fit` when given (for example the
    nonzero entries only) and the plotted points otherwise. Both axes share one
    range, from `lower_limit` (or 5% below the data) to 5% above the data.
    """
    x_fit = x if x_fit is None else x_fit
    y_fit = y if y_fit is None else y_fit
    r_val = stats.spearmanr(x_fit, y_fit)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RankWarning)
        m, b = np.polyfit(x_fit, y_fit, 1)

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1, bins='log')

    data_max = float(max(x.max(), y.max()))
    if lower_limit is not None:
        ax_min, ax_max = lower_limit, data_max * 1.05
    else:
        ax_min = float(min(x.min(), y.min())) * 1.05
        ax_max = data_max * 1.05
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    xr_ = np.linspace(x_fit.min(), x_fit.max(), 100)
    ax.plot(xr_, m * xr_ + b, color=reg_color, lw=reg_lw, ls='-')

    if show_text:
        if show_eq:
            txt = f'$r={r_val:.2f}$\n$y={m:.2f}x{b:+.2f}$'
        else:
            txt = f'$r={r_val:.2f}$'
        ax.text(
            text_loc[0],
            text_loc[1],
            txt,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontweight=text_fontweight,
            color='black',
            zorder=100,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
        )

    if x_label:
        ax.set_xlabel(x_label, labelpad=axis_labelpad)
    if y_label:
        ax.set_ylabel(y_label, labelpad=axis_labelpad)
    if title:
        ax.set_title(title)
    ax.patch.set_alpha(0)

    if show_cbar:
        cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
        cb = plt.colorbar(hb, cax=cax)
        vmin, vmax = hb.get_clim()
        if compact_cbar:
            cb.set_ticks([vmin, vmax])
            cb.set_ticklabels([compact_count_label(vmin), compact_count_label(vmax)])
            if cbar_label:
                cb.set_label(cbar_label, rotation=90, labelpad=cbar_labelpad)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0, which='both')
        else:
            clean_colorbar(cb, vmin, vmax, label=cbar_label, labelpad=cbar_labelpad)

    modify_axis_spines(
        ax,
        which=['x', 'y'],
        xticks_main=[round(ax_min, x_round), round(ax_max, x_round)],
        yticks_main=[round(ax_min, y_round), round(ax_max, y_round)],
    )
    return ax


def plot_group_regressions(
    df,
    ax=None,
    *,
    x_col,
    y_col,
    group_col,
    groups=None,
    group_labels=None,
    y_label_fmt='{}',
    x_label=None,
    fit_linear=True,
    x_range=None,
    y_lim=None,
    y_ticks=None,
    split_col=None,
    split_colors=('slategray', 'slategray'),
    scatter_color='slategray',
    scatter_size=20,
    scatter_alpha=0.8,
    line_width=2,
    line_color='indianred',
    show_rho=True,
    rho_pos=(0.95, 0.95),
    figsize=None,
    savefig=False,
    save_name='group_regressions.svg',
    fontsize=None,
    font_scale=1.0,
):
    """One scatter panel per group of a tidy table, with a linear fit and the
    Spearman rho. With `ax` only the first group is drawn into it; otherwise the
    groups fill a grid two panels wide. `split_col` (0/1) recolours points
    within a panel."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        group_labels = group_labels or {}
        if groups is None:
            groups = list(dict.fromkeys(df[group_col].dropna()))
        if len(groups) == 0:
            return None, None

        if ax is not None:
            fig = ax.figure
            axes = np.array([[ax]])
            groups = groups[:1]
        else:
            ncols = min(2, len(groups))
            nrows = int(np.ceil(len(groups) / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize or (5 * ncols, 4 * nrows), squeeze=False
            )

        axes_flat = axes.flatten()
        lo, hi = x_range if x_range is not None else (df[x_col].min(), df[x_col].max())
        x_eval = np.linspace(lo, hi, 200)
        last_idx = -1
        for i, group in enumerate(groups):
            ax_i, last_idx = axes_flat[i], i
            sub_df = df[df[group_col] == group].dropna(subset=[x_col, y_col])
            if len(sub_df) == 0:
                ax_i.axis('off')
                continue
            x, y = sub_df[x_col].values, sub_df[y_col].values
            c = (
                np.where(sub_df[split_col] == 1, split_colors[1], split_colors[0])
                if (split_col is not None and split_col in sub_df.columns)
                else scatter_color
            )
            ax_i.scatter(
                x, y, c=c, s=scatter_size, alpha=scatter_alpha, edgecolors='none', zorder=2
            )
            if fit_linear:
                coef = np.polyfit(x, y, 1)
                ax_i.plot(
                    x_eval,
                    np.polyval(coef, x_eval),
                    color=line_color,
                    lw=line_width,
                    ls='--',
                    zorder=3,
                )
            if show_rho:
                ax_i.text(
                    rho_pos[0],
                    rho_pos[1],
                    fr"$\rho$ = {stats.spearmanr(x, y)[0]:.2f}",
                    transform=ax_i.transAxes,
                    ha='right',
                    va='top',
                )
            ax_i.set_xlabel(x_label or x_col)
            ax_i.set_ylabel(y_label_fmt.format(group_labels.get(group, group)))
            ax_i.set_xlim([lo, hi])
            if y_lim is not None:
                ax_i.set_ylim(y_lim)
            ax_i.patch.set_alpha(0)
            modify_axis_spines(ax_i, which=['x', 'y'], xticks_main=[lo, hi], yticks_main=y_ticks)
        for j in range(last_idx + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        if len(fig.axes) > 1:
            fig.tight_layout()
        if savefig:
            save_figure(fig, save_name)

        return fig, axes


def plot_group_regression_grid(
    df,
    y_vars,
    *,
    x_var,
    group_col='Group',
    y_labels=None,
    figsize=None,
    colors=None,
    s=3,
    alpha=0.6,
    reg_color='indianred',
    reg_lw=2,
    precisions=None,
    text_box_alpha=0.5,
    group_order=None,
    label_fn=None,
    pooled_label='Global',
    savefig=False,
    save_name='group_grid.svg',
    fontsize=None,
    font_scale=1.0,
):
    """Grid of scatter-plus-fit panels: rows are variables, columns are the groups
    of `group_col` followed by all groups pooled. Columns follow `group_order`
    (unlisted groups appended), else first appearance."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        groups_seen = list(dict.fromkeys(df[group_col].dropna().values))
        if group_order is not None:
            groups = [c for c in group_order if c in groups_seen] + [
                c for c in groups_seen if c not in group_order
            ]
        else:
            groups = groups_seen
        plot_cols = groups + [pooled_label]

        if colors is None:
            palette = sns.color_palette("inferno", len(groups))
            colors = {c: palette[i] for i, c in enumerate(groups)}
        elif isinstance(colors, str):
            colors = {c: colors for c in groups}

        if pooled_label not in colors:
            colors[pooled_label] = colors[groups[0]]

        n_rows, n_cols = len(y_vars), len(plot_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (3.5 * n_cols, 3 * n_rows), squeeze=False
        )

        for row_idx, y_var in enumerate(y_vars):
            p = precisions.get(y_var, 2) if isinstance(precisions, dict) else (precisions or 2)
            y_clean = df[y_var].dropna()
            x_clean = df[x_var].dropna()
            if len(y_clean) == 0 or len(x_clean) == 0:
                continue
            y_min, y_max = y_clean.min(), y_clean.max()
            x_min, x_max = x_clean.min(), x_clean.max()

            for col_idx, col_name in enumerate(plot_cols):
                ax = axes[row_idx, col_idx]

                if col_name == pooled_label:
                    sub_df = df.dropna(subset=[y_var, x_var])
                    plot_regression_layer(
                        ax,
                        sub_df[x_var],
                        sub_df[y_var],
                        color=colors[pooled_label],
                        reg_color=reg_color,
                        s=s,
                        alpha=alpha,
                        reg_lw=reg_lw,
                        text_box_alpha=text_box_alpha,
                    )
                else:
                    sub_df = df[df[group_col] == col_name].dropna(subset=[y_var, x_var])
                    plot_regression_layer(
                        ax,
                        sub_df[x_var],
                        sub_df[y_var],
                        color=colors.get(col_name, 'slategray'),
                        reg_color=reg_color,
                        s=s,
                        alpha=alpha,
                        reg_lw=reg_lw,
                        text_box_alpha=text_box_alpha,
                    )

                ax.patch.set_alpha(0)
                modify_axis_spines(
                    ax,
                    which=['x', 'y'],
                    xticks_main=[int(x_min), int(x_max)],
                    yticks_main=[np.round(y_min, p), np.round(y_max, p)],
                )

                if row_idx < n_rows - 1:
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel(x_var.capitalize())
                if col_idx > 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                else:
                    ax.set_ylabel(y_labels[row_idx] if y_labels else y_var)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.15, hspace=0.15)

        for col_idx, col_name in enumerate(plot_cols):
            label = label_fn(col_name) if (label_fn and col_name != pooled_label) else str(col_name)
            axes[0, col_idx].set_title(label, fontweight='bold', pad=10)

        if savefig:
            save_figure(fig, save_name)
        return fig, axes


def plot_binned_overlay(
    x,
    y,
    strat,
    ax=None,
    *,
    n_bins=3,
    qcut=True,
    x_label=None,
    y_label=None,
    strat_label=None,
    palette='viridis',
    s=20,
    alpha=0.6,
    scatter_ec='k',
    scatter_ew=0.5,
    median_lw=3,
    median_ls='-',
    median_extents=(25, 75),
    median_ec='k',
    median_ew=1.0,
    figsize=(6, 5),
    x_precision=2,
    y_precision=4,
    show_legend=True,
    legend_kwargs=None,
    show_cbar=False,
    cbar_ax=None,
    cbar_label=None,
    cbar_inset_bounds=(0.75, 0.55, 0.05, 0.4),
    savefig=False,
    save_name='binned_overlay.svg',
    title=None,
    fontsize=None,
    font_scale=1.0,
):
    """Scatter of y against x split into bins of a third variable, each bin with
    crosshairs through its medians spanning `median_extents` percentiles (or
    'minmax'). Bins are quantiles with `qcut`, equal-width otherwise; they can be
    keyed by a legend, a stepped colorbar, or both."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        x = np.asarray(x)
        y = np.asarray(y)
        f_val = np.asarray(strat)
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(f_val)
        x_clean, y_clean, f_clean = x[mask], y[mask], f_val[mask]
        if len(x_clean) == 0:
            raise ValueError("No valid overlapping data points found.")

        df = pd.DataFrame({'x': x_clean, 'y': y_clean, 'f': f_clean})
        df['bin'] = (
            pd.qcut(df['f'], q=n_bins, duplicates='drop') if qcut else pd.cut(df['f'], bins=n_bins)
        )

        unique_bins = df['bin'].cat.categories
        n_actual_bins = len(unique_bins)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cmap = plt.get_cmap(palette)
        colors = [cmap(i) for i in np.linspace(0.15, 0.85, n_actual_bins)]
        x_min, x_max = x_clean.min(), x_clean.max()
        y_min, y_max = y_clean.min(), y_clean.max()
        legend_handles = []

        for i, b in enumerate(unique_bins):
            sub = df[df['bin'] == b]
            color = colors[i]
            bin_label = f"{b.left:.2g}-{b.right:.2g}"
            ax.scatter(
                sub['x'],
                sub['y'],
                color=color,
                alpha=alpha,
                s=s,
                zorder=1,
                edgecolors=scatter_ec if scatter_ec else 'none',
                linewidths=scatter_ew if scatter_ec else 0,
            )
            if len(sub) > 0:
                x_med = np.median(sub['x'])
                y_med = np.median(sub['y'])
                if median_extents == 'minmax':
                    x_low, x_high = sub['x'].min(), sub['x'].max()
                    y_low, y_high = sub['y'].min(), sub['y'].max()
                elif isinstance(median_extents, (list, tuple)) and len(median_extents) == 2:
                    x_low, x_high = np.percentile(sub['x'], median_extents)
                    y_low, y_high = np.percentile(sub['y'], median_extents)
                else:
                    x_low, x_high = np.percentile(sub['x'], [25, 75])
                    y_low, y_high = np.percentile(sub['y'], [25, 75])
                if median_lw > 0:
                    if median_ec:
                        out_lw = median_lw + (median_ew * 2)
                        ax.plot(
                            [x_low, x_high],
                            [y_med, y_med],
                            color=median_ec,
                            ls=median_ls,
                            lw=out_lw,
                            zorder=2.9,
                        )
                        ax.plot(
                            [x_med, x_med],
                            [y_low, y_high],
                            color=median_ec,
                            ls=median_ls,
                            lw=out_lw,
                            zorder=2.9,
                        )
                    ax.plot(
                        [x_low, x_high],
                        [y_med, y_med],
                        color=color,
                        ls=median_ls,
                        lw=median_lw,
                        zorder=3,
                    )
                    ax.plot(
                        [x_med, x_med],
                        [y_low, y_high],
                        color=color,
                        ls=median_ls,
                        lw=median_lw,
                        zorder=3,
                    )
                ax.scatter(
                    [x_med],
                    [y_med],
                    color=color,
                    edgecolor=median_ec if median_ec else 'white',
                    s=s * 4,
                    zorder=4,
                    marker='o',
                    linewidths=median_ew if median_ec else 1.0,
                )
            legend_handles.append(mpatches.Patch(color=color, label=bin_label))

        ax.patch.set_alpha(0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title, fontweight='bold', pad=10)
        modify_axis_spines(
            ax,
            which=['x', 'y'],
            xticks_main=[np.round(x_min, x_precision), np.round(x_max, x_precision)],
            yticks_main=[np.round(y_min, y_precision), np.round(y_max, y_precision)],
        )
        fs = fontsize or 7
        if show_legend:
            if legend_kwargs is None:
                legend_kwargs = {}
            ax.legend(
                handles=legend_handles,
                **{
                    **{'frameon': False, 'loc': 'upper right', 'title': strat_label},
                    **legend_kwargs,
                },
            )

        if show_cbar:
            edges_lo = np.array(
                [float(f_clean.min()) if np.isinf(b.left) else float(b.left) for b in unique_bins]
            )
            edges_hi = np.array(
                [float(f_clean.max()) if np.isinf(b.right) else float(b.right) for b in unique_bins]
            )
            boundaries = np.concatenate([[edges_lo[0]], edges_hi])
            tick_vals = (edges_lo + edges_hi) / 2
            lc = ListedColormap(colors)
            bn = BoundaryNorm(boundaries, len(colors))
            sm = plt.cm.ScalarMappable(cmap=lc, norm=bn)
            sm.set_array([])
            _cax = cbar_ax if cbar_ax is not None else ax.inset_axes(list(cbar_inset_bounds))
            cb = fig.colorbar(sm, cax=_cax, ticks=tick_vals)
            cb.set_ticklabels([f'{lo:.0f}-{hi:.0f}' for lo, hi in zip(edges_lo, edges_hi)])
            cb.set_label(cbar_label or strat_label, fontsize=fs)
            cb.outline.set_visible(False)
            cb.ax.tick_params(size=0, labelsize=fs)

        if savefig:
            save_figure(fig, save_name)

        return fig, ax

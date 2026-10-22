"""Distributions: rainclouds, violins and ridgelines."""

import colorsys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator
from vizard.annotate import CustomPValueFormat
from vizard.style import font_rc, add_styled_colorbar, modify_axis_spines, save_figure

# ===== raincloud layers =====


def _levels(values, order):
    if order is not None:
        return list(order)
    values = pd.Series(values)
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    levels = list(values.dropna().unique())
    return sorted(levels) if pd.api.types.is_numeric_dtype(values) else levels


def _desaturate(color, prop):
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(h, l, s * prop)


# matplotlib's qualitative maps are read colour by colour; any other colormap is
# sampled evenly with its two ends left out
qualitative_cmaps = {
    'tab10',
    'tab20',
    'tab20b',
    'tab20c',
    'Set1',
    'Set2',
    'Set3',
    'Accent',
    'Paired',
    'Pastel1',
    'Pastel2',
    'Dark2',
}


def _palette_colors(palette, levels):
    n = len(levels)
    if isinstance(palette, dict):
        return [palette[level] for level in levels]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        if palette in qualitative_cmaps:
            return [cmap.colors[i % len(cmap.colors)] for i in range(n)]
        return [cmap(v) for v in np.linspace(0, 1, n + 2)[1:-1]]
    if palette is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return [cycle[i % len(cycle)] for i in range(n)]
    return [palette[i % len(palette)] for i in range(n)]


def _half_violins(
    ax,
    data,
    x,
    y,
    hue=None,
    order=None,
    hue_order=None,
    palette=None,
    color=None,
    bw=0.2,
    cut=0.0,
    scale='area',
    width=0.7,
    offset=0.15,
    linewidth=None,
    orient='v',
    saturation=0.75,
    **kws,
):
    """One half-violin per category (and hue level), drawn on the low side of
    the category position and shifted further down by `offset`.

    Densities are Gaussian KDEs with `bw` as the bandwidth factor, evaluated
    from `cut` bandwidths below the smallest observation to as far above the
    largest. With `scale='area'` they share one scale per category, so the
    widest spans `width` / 2; 'width' scales each to that width, and 'count'
    scales by the number of observations. Hue colours are desaturated by
    `saturation`, and outlines take a grey darker than the lightest fill.
    """
    horizontal = orient in ('h', 'y')
    cat_col, val_col = (y, x) if horizontal else (x, y)
    cat_levels = _levels(data[cat_col], order)
    keys = [cat_col] + ([hue] if hue is not None else [])

    if hue is not None:
        hue_levels = _levels(data[hue], hue_order)
        colors = {
            level: _desaturate(c, saturation)
            for level, c in zip(hue_levels, _palette_colors(palette, hue_levels))
        }
        lightness = [
            colorsys.rgb_to_hls(*rgb)[1] for rgb in np.unique(list(colors.values()), axis=0)
        ]
        gray = (min(lightness) * 0.6,) * 3
    else:
        if color is not None:
            cat_colors = [color] * len(cat_levels)
        elif palette is not None:
            cat_colors = [mcolors.to_rgb(c) for c in _palette_colors(palette, cat_levels)]
        else:
            cat_colors = [
                mcolors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            ] * len(cat_levels)
        colors = dict(zip(cat_levels, cat_colors))
        gray = (colorsys.rgb_to_hls(*mcolors.to_rgb(color))[1] * 0.6,) * 3 if color else '0.3'
    linewidth = mpl.rcParams['lines.linewidth'] if linewidth is None else linewidth

    violins = []
    for key, group in data.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        values = group[val_col].dropna()
        if values.size > 1 and np.unique(values).size > 1:
            kde = stats.gaussian_kde(np.asarray(values, dtype=float), bw)
            bw_used = kde.factor * np.asarray(values, dtype=float).std(ddof=1)
            support = np.linspace(values.min() - bw_used * cut, values.max() + bw_used * cut, 100)
            density = kde.evaluate(support)
        else:
            support, density = np.unique(values), np.ones(min(values.size, 1))
        violins.append(dict(key=key, support=support, density=density, n=values.size))

    peak = {id(v): v['density'].max() if v['density'].size > 1 else 0 for v in violins}
    for v in violins:
        if v['density'].size <= 1:
            continue
        if scale == 'width':
            v['density'] = v['density'] / peak[id(v)]
        elif scale == 'count':
            same_cat = [u for u in violins if u['key'][0] == v['key'][0]] if hue else violins
            v['density'] = v['density'] / peak[id(v)] * v['n'] / max(u['n'] for u in same_cat)
        else:
            same_cat = [u for u in violins if u['key'][0] == v['key'][0]] if hue else violins
            top = max(peak[id(u)] for u in same_cat)
            if top > 0:
                v['density'] = v['density'] / top

    fill = ax.fill_between if horizontal else ax.fill_betweenx
    for v in violins:
        if v['support'].size == 0:
            continue
        center = cat_levels.index(v['key'][0])
        face = colors[v['key'][1]] if hue is not None else colors[v['key'][0]]
        if v['support'].size == 1:
            half = v['density'][0] * width / 2
            ends = ([center - half, center + half], [v['support'][0]] * 2)
            ax.plot(*(ends[::-1] if horizontal else ends), color=gray, linewidth=linewidth)
            continue
        base = np.full(v['support'].size, center - offset)
        fill(
            v['support'],
            base - v['density'] * width / 2,
            base,
            facecolor=face,
            **{**kws, 'edgecolor': gray, 'linewidth': linewidth},
        )

    ticks = range(len(cat_levels))
    if horizontal:
        ax.set_yticks(ticks)
        ax.set_yticklabels(cat_levels)
        ax.set_ylim(len(cat_levels) - 0.5, -0.5)
        ax.yaxis.grid(False)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(cat_levels)
        ax.set_xlim(-0.5, len(cat_levels) - 0.5)
        ax.xaxis.grid(False)
    return ax


def _strip(
    ax,
    data,
    x,
    y,
    hue=None,
    order=None,
    hue_order=None,
    palette=None,
    orient=None,
    dodge=False,
    move=0.0,
    width=0.8,
    size=5,
    edgecolor='gray',
    linewidth=0,
    **kws,
):
    """Jittered points through seaborn's stripplot, shifted by `move` along the
    category axis, with hue groups dodged to match boxes `width` wide.

    seaborn dodges as though boxes were 0.8 wide, so each group's offset (the
    median shift of its points) is rescaled while the jitter is kept.
    """
    horizontal = orient in ('h', 'y')
    if palette is not None and hue is None and isinstance(x, str) and isinstance(y, str):
        # colour by category through hue, as seaborn requires, without a legend
        hue = y if horizontal else x
        kws.setdefault('legend', False)
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == 'gray':
        edgecolor = '0.3'

    n_before = len(ax.collections)
    sns.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        order=order,
        hue_order=hue_order,
        dodge=dodge,
        orient=orient,
        palette=palette,
        size=size,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax,
        **kws,
    )

    axis = 1 if horizontal else 0
    for points in ax.collections[n_before:]:
        offsets = points.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue
        offsets = np.asarray(offsets)
        if dodge and width != 0.8:
            pos = offsets[:, axis]
            centers = np.round(pos)
            shift = np.median(pos - centers)
            offsets[:, axis] = centers + shift * width / 0.8 + (pos - centers - shift)
        if move != 0:
            offsets[:, axis] += move
        points.set_offsets(offsets)
    return ax


def plot_rainclouds(
    x=None,
    y=None,
    hue=None,
    data=None,
    num_samples=1000,
    swarm_sample=False,
    order=None,
    hue_order=None,
    plot_legend=False,
    legend_kwargs=None,
    orient="v",
    width_viol=0.7,
    width_box=0.15,
    clouds=True,
    boxes=True,
    rain=True,
    palette="Set2",
    bw=0.2,
    linewidth=1,
    cut=0.0,
    scale="area",
    jitter=0.05,
    move=0.0,
    offset=None,
    point_size=3,
    ax=None,
    pointplot=False,
    alpha=None,
    dodge_boxes=True,
    linecolor='red',
    violin_width=None,
    violin_linewidth=None,
    violin_cut=None,
    box_linewidth=1.5,
    box_show_outliers=False,
    box_outlier_marker='d',
    box_outlier_size=3,
    box_strict_style=True,
    rain_show_outliers=True,
    fontsize=None,
    font_scale=1.0,
    **kwargs,
):
    """Raincloud plot: a half-violin, a box and jittered points per category.

    After Allen et al. (2019). Extra keywords prefixed with cloud_,
    box_, rain_ or point_ go to the matching layer. With `box_strict_style` the
    box, whiskers and median share each box's own colour and caps are dropped.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if legend_kwargs is None:
            legend_kwargs = {}

        if violin_width is not None:
            width_viol = violin_width
        if violin_linewidth is not None:
            linewidth = violin_linewidth
        if violin_cut is not None:
            cut = violin_cut

        if orient == 'h':
            x, y = y, x
        if ax is None:
            ax = plt.gca()

        ax.patch.set_alpha(0)

        if offset is None:
            offset = max(width_box / 1.8, 0.15) + 0.05
        n_plots = 3
        boxcolor = "black"
        boxprops = {'facecolor': 'none', "zorder": 10}

        kwcloud = dict()
        kwbox = dict(saturation=1, whiskerprops={'linewidth': 2, "zorder": 10})
        kwrain = dict(zorder=0, edgecolor=None)
        kwpoint = dict(capsize=0.0, errwidth=0.0, zorder=20)

        if box_strict_style:
            kwbox['showcaps'] = False
            kwbox['whiskerprops']['linewidth'] = box_linewidth
            # median colour is matched to its box after drawing
            kwbox['medianprops'] = {'linewidth': box_linewidth, 'zorder': 10}
            kwbox['boxprops'] = {'linewidth': box_linewidth, 'zorder': 10, 'facecolor': 'none'}
            boxprops = kwbox['boxprops']

            if not box_show_outliers:
                kwbox['showfliers'] = False
            else:
                kwbox['showfliers'] = True
                kwbox['flierprops'] = dict(
                    marker=box_outlier_marker,
                    markersize=box_outlier_size,
                    markerfacecolor='gray',
                    markeredgecolor='none',
                    zorder=10,
                )

        box_alpha_val = None

        for key, value in kwargs.items():
            if "cloud_" in key:
                kwcloud[key.replace("cloud_", "")] = value
            elif "box_" in key:
                k = key.replace("box_", "")
                if k == 'alpha':
                    box_alpha_val = value
                else:
                    kwbox[k] = value
            elif "rain_" in key:
                kwrain[key.replace("rain_", "")] = value
            elif "point_" in key:
                kwpoint[key.replace("point_", "")] = value
            else:
                kwcloud[key] = value

        if swarm_sample:
            swarm_df = data.sample(num_samples)
        else:
            swarm_df = data.copy() if hasattr(data, 'copy') else data

        if not rain_show_outliers and isinstance(swarm_df, pd.DataFrame):
            try:
                val_col = y if orient == 'v' else x
                cat_col = x if orient == 'v' else y

                if isinstance(val_col, str) and val_col in swarm_df.columns:
                    groups = []
                    if isinstance(cat_col, str) and cat_col in swarm_df.columns:
                        groups.append(cat_col)
                    if isinstance(hue, str) and hue in swarm_df.columns:
                        groups.append(hue)

                    def filter_outliers(g):
                        if len(g) < 2:
                            return g
                        q1 = g[val_col].quantile(0.25)
                        q3 = g[val_col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        return g[(g[val_col] >= lower) & (g[val_col] <= upper)]

                    if groups:
                        swarm_df = swarm_df.groupby(groups, group_keys=False).apply(filter_outliers)
                    else:
                        swarm_df = filter_outliers(swarm_df)
            except Exception as e:
                print(f"Warning: Could not filter outliers from swarm plot: {e}")

        if clouds:
            _cloud_alpha = kwcloud.pop('alpha', None)
            _n_col_before = len(ax.collections)
            _half_violins(
                ax,
                data,
                x,
                y,
                hue=hue,
                order=order,
                hue_order=hue_order,
                orient=orient,
                width=width_viol,
                palette=palette,
                bw=bw,
                linewidth=linewidth,
                cut=cut,
                scale=scale,
                offset=offset,
                **kwcloud,
            )
            if _cloud_alpha is not None:
                for _col in ax.collections[_n_col_before:]:
                    _fc = np.array(_col.get_facecolor())
                    if _fc.ndim == 2 and _fc.shape[1] == 4 and len(_fc) > 0:
                        _fc[:, 3] = _cloud_alpha
                        _col.set_facecolor(_fc)

        if 'boxprops' in kwbox:
            del kwbox['boxprops']

        if boxes:
            sns.boxplot(
                x=x,
                y=y,
                hue=hue,
                data=data,
                orient=orient,
                width=width_box,
                order=order,
                hue_order=hue_order,
                dodge=dodge_boxes,
                color=boxcolor,
                boxprops=boxprops,
                palette=palette,
                ax=ax,
                **kwbox,
            )

            # seaborn draws box lines in its own grey; recolour each group to its box
            if box_strict_style:
                has_fliers = kwbox.get('showfliers', True)
                lines_per_box = 2 + 1 + (1 if has_fliers else 0)

                boxes_patches = ax.patches
                box_lines = [l for l in ax.lines if l.get_zorder() == 10]

                if len(boxes_patches) > 0:
                    expected_lines = len(boxes_patches) * lines_per_box
                    if len(box_lines) >= expected_lines:
                        relevant_lines = box_lines[-expected_lines:]
                        for i, patch in enumerate(boxes_patches):
                            col = patch.get_edgecolor()
                            start = i * lines_per_box
                            end = start + lines_per_box
                            group_lines = relevant_lines[start:end]
                            for line in group_lines:
                                line.set_color(col)
                                line.set_linewidth(box_linewidth)
                                if has_fliers and line == group_lines[-1]:
                                    line.set_marker(box_outlier_marker)
                                    line.set_markersize(box_outlier_size)
                                    line.set_markeredgecolor(col)
                                    line.set_markerfacecolor(col)

            if box_alpha_val is not None:
                for patch in ax.patches:
                    if patch.get_zorder() == 10:
                        patch.set_alpha(box_alpha_val)

        if not alpha is None:
            _ = plt.setp(ax.collections + ax.artists, alpha=alpha)

        if rain:
            ax = _strip(
                ax,
                swarm_df,
                x,
                y,
                hue=hue,
                orient=orient,
                dodge=dodge_boxes,
                order=order,
                hue_order=hue_order,
                palette=palette,
                move=move,
                size=point_size,
                jitter=jitter,
                width=width_box,
                **kwrain,
            )

        if pointplot:
            n_plots = 4
            if not hue is None:
                sns.pointplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=data,
                    orient=orient,
                    order=order,
                    hue_order=hue_order,
                    dodge=width_box / 2.0,
                    palette=palette,
                    ax=ax,
                    **kwpoint,
                )
            else:
                sns.pointplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=data,
                    color=linecolor,
                    orient=orient,
                    order=order,
                    hue_order=hue_order,
                    dodge=width_box / 2.0,
                    ax=ax,
                    **kwpoint,
                )

        if (not hue is None) and (plot_legend):
            handles, labels = ax.get_legend_handles_labels()
            default_legend = {
                'bbox_to_anchor': (1.05, 1),
                'loc': 2,
                'borderaxespad': 0.0,
                'title': str(hue),
            }
            final_legend_kwargs = {**default_legend, **{'frameon': False}, **legend_kwargs}
            plt.legend(
                handles[0 : len(labels) // n_plots],
                labels[0 : len(labels) // n_plots],
                **final_legend_kwargs,
            )

        if orient == "h":
            ylim = list(ax.get_ylim())
            ylim[-1] -= (width_box + width_viol) / 4.0
            _ = ax.set_ylim(ylim)
        elif orient == "v":
            xlim = list(ax.get_xlim())
            xlim[-1] -= (width_box + width_viol) / 4.0
            _ = ax.set_xlim(xlim)

        return ax


def plot_posterior_ridge(
    posteriors,
    param_name,
    x_metric,
    ax=None,
    bin_size=5.0,
    overlap=1.5,
    figsize=(6, 6),
    cmap='inferno',
    cmap_range=None,
    alpha=0.6,
    linewidth=1.0,
    bandwidth=None,
    bw_adjust=1.0,
    fit_bins_to_data=False,
    cut=0,
    clip=None,
    sample_quantiles=None,
    ridge_halo=False,
    plot_trend=False,
    trend_color='black',
    trend_lw=1.5,
    trend_linestyle='-',
    trend_alpha=1.0,
    trend_offset=0.0,
    trend_statistic='peak',
    mark_median=False,
    median_mark_color='white',
    median_mark_lw=1.5,
    median_mark_linestyle='-',
    median_mark_alpha=1.0,
    median_mark_offset=0.0,
    median_statistic='median',
    title=None,
    xlabel='Parameter Value',
    ylabel='Covariate Bin',
    orientation='horizontal',
    tick_precision=0,
    param_precision=2,
    xlim=None,
    ylim=None,
    metric_color_min=None,
    metric_color_max=None,
    savefig=False,
    suffix='',
    labelpad=5,
    fontsize=None,
    font_scale=1.0,
):
    """Ridgeline of one posterior parameter per bin of a subject covariate.

    `posteriors` is a (subject, sample, theta) DataArray and `x_metric` the
    covariate per subject. Each bin's samples are pooled into a KDE drawn on
    its own baseline and coloured by the bin centre; `plot_trend` joins the
    ridges' peaks (or means or medians) and `mark_median` draws a tick at each
    ridge's median.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if not isinstance(posteriors, xr.DataArray):
            raise ValueError('posteriors must be an xarray.DataArray.')
        da = posteriors.sel(theta=param_name)
        metric_vals = x_metric.values if hasattr(x_metric, 'values') else np.array(x_metric)

        min_metric, max_metric = metric_vals.min(), metric_vals.max()

        if fit_bins_to_data:
            bins = np.arange(min_metric, max_metric + bin_size * 0.1, bin_size)
            if bins[-1] < max_metric:
                bins = np.append(bins, bins[-1] + bin_size)
            # The trailing edge is a multiple of bin_size and can overshoot the
            # sample, which then labels the last ridge over an age nobody has.
            if bins[-1] > max_metric > bins[-2]:
                bins[-1] = max_metric
        else:
            start = np.floor(min_metric / bin_size) * bin_size
            end = np.ceil(max_metric / bin_size) * bin_size
            if end == start:
                end += bin_size
            bins = np.arange(start, end + bin_size * 0.1, bin_size)
            if bins[-1] < max_metric:
                bins = np.append(bins, bins[-1] + bin_size)

        valid_bins, global_bounds = [], []

        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (metric_vals >= bins[i]) & (metric_vals <= bins[i + 1])
            else:
                mask = (metric_vals >= bins[i]) & (metric_vals < bins[i + 1])

            if mask.sum() > 0:
                data = da.isel(subject=mask).values.flatten()
                kde = gaussian_kde(data, bw_method=bandwidth)
                kde.set_bandwidth(kde.factor * bw_adjust)
                bw = np.sqrt(kde.covariance[0, 0])
                # Bounds only the drawn support; the KDE and median still use
                # the full sample. Raw min/max are sampler stragglers.
                if sample_quantiles is not None:
                    lo_edge, hi_edge = np.quantile(data, sample_quantiles)
                else:
                    lo_edge, hi_edge = data.min(), data.max()
                lb = lo_edge - cut * bw
                ub = hi_edge + cut * bw
                if clip is not None:
                    if clip[0] is not None:
                        lb = max(lb, clip[0])
                    if clip[1] is not None:
                        ub = min(ub, clip[1])
                valid_bins.append((bins[i], bins[i + 1], data, kde, lb, ub))
                global_bounds.extend([lb, ub])

        if not valid_bins:
            raise ValueError("No valid bins found.")

        global_min, global_max = min(global_bounds), max(global_bounds)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        c_min = metric_color_min if metric_color_min is not None else min_metric
        c_max = metric_color_max if metric_color_max is not None else max_metric
        norm = mcolors.Normalize(vmin=c_min, vmax=c_max)

        if cmap_range:
            base_cmap = plt.get_cmap(cmap)
            cmap_obj = mcolors.LinearSegmentedColormap.from_list(
                f'trunc_{base_cmap.name}', base_cmap(np.linspace(cmap_range[0], cmap_range[1], 256))
            )
        else:
            cmap_obj = plt.get_cmap(cmap)

        ticks, ticklabels, trend_points = [], [], []
        for i, (l, u, data, kde, lb, ub) in enumerate(valid_bins):
            grid_local = np.linspace(lb, ub, 500)
            pdf = kde(grid_local)
            pdf_max = pdf.max()
            pdf_norm = pdf / pdf_max
            baseline = i
            color = cmap_obj(norm((l + u) / 2))

            if trend_statistic == 'mean':
                stat_val = np.mean(data)
            elif trend_statistic == 'median':
                stat_val = np.median(data)
            else:
                stat_val = grid_local[np.argmax(pdf)]
            trend_points.append(stat_val)

            if orientation == 'horizontal':
                ax.fill_between(
                    grid_local,
                    baseline,
                    baseline + overlap * pdf_norm,
                    color=color,
                    alpha=alpha,
                    zorder=len(valid_bins) - i,
                )
                if ridge_halo:
                    ax.plot(
                        grid_local,
                        baseline + overlap * pdf_norm,
                        color='white',
                        lw=linewidth + 0.5,
                        zorder=len(valid_bins) - i + 0.1,
                    )
                ax.plot(
                    grid_local,
                    baseline + overlap * pdf_norm,
                    color=color,
                    lw=linewidth,
                    alpha=1.0,
                    zorder=len(valid_bins) - i + 0.2,
                )
                if mark_median:
                    med = np.mean(data) if median_statistic == 'mean' else np.median(data)
                    med_h = (kde(med) / pdf_max).item()
                    ax.plot(
                        [med, med],
                        [baseline, baseline + overlap * med_h - median_mark_offset],
                        color=median_mark_color,
                        lw=median_mark_lw,
                        linestyle=median_mark_linestyle,
                        alpha=median_mark_alpha,
                        zorder=len(valid_bins) - i + 0.3,
                    )
            else:
                ax.fill_betweenx(
                    grid_local,
                    baseline,
                    baseline + overlap * pdf_norm,
                    color=color,
                    alpha=alpha,
                    zorder=len(valid_bins) - i,
                )
                if ridge_halo:
                    ax.plot(
                        baseline + overlap * pdf_norm,
                        grid_local,
                        color='white',
                        lw=linewidth + 0.5,
                        zorder=len(valid_bins) - i + 0.1,
                    )
                ax.plot(
                    baseline + overlap * pdf_norm,
                    grid_local,
                    color=color,
                    lw=linewidth,
                    alpha=1.0,
                    zorder=len(valid_bins) - i + 0.2,
                )
                if mark_median:
                    med = np.mean(data) if median_statistic == 'mean' else np.median(data)
                    med_h = (kde(med) / pdf_max).item()
                    ax.plot(
                        [baseline, baseline + overlap * med_h - median_mark_offset],
                        [med, med],
                        color=median_mark_color,
                        lw=median_mark_lw,
                        linestyle=median_mark_linestyle,
                        alpha=median_mark_alpha,
                        zorder=len(valid_bins) - i + 0.3,
                    )
            ticks.append(baseline)
            ticklabels.append(f'{l:.{tick_precision}f}-{u:.{tick_precision}f}')

        if plot_trend:
            trend_heights = np.arange(len(valid_bins)) + overlap + trend_offset
            if orientation == 'horizontal':
                ax.plot(
                    trend_points,
                    trend_heights,
                    color=trend_color,
                    lw=trend_lw,
                    linestyle=trend_linestyle,
                    alpha=trend_alpha,
                    zorder=len(valid_bins) + 1,
                )
            else:
                ax.plot(
                    trend_heights,
                    trend_points,
                    color=trend_color,
                    lw=trend_lw,
                    linestyle=trend_linestyle,
                    alpha=trend_alpha,
                    zorder=len(valid_bins) + 1,
                )

        if title:
            ax.set_title(title)

        if orientation == 'horizontal':
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_xlabel(xlabel, labelpad=labelpad)
            ax.set_ylabel(ylabel, labelpad=labelpad)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            final_xlim = xlim if xlim else (global_min, global_max)
            ax.set_xlim(final_xlim)
            ax.patch.set_alpha(0)
            modify_axis_spines(ax, which=['x'], xticks_main=np.round(final_xlim, param_precision))
        else:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, rotation=45)
            ax.set_xlabel(ylabel, labelpad=labelpad)
            ax.set_ylabel(xlabel, labelpad=labelpad)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', length=0)
            final_ylim = ylim if ylim else (global_min, global_max)
            ax.set_ylim(final_ylim)
            ax.patch.set_alpha(0)
            modify_axis_spines(ax, which=['y'], yticks_main=np.round(final_ylim, param_precision))

        if savefig:
            save_figure(
                fig,
                (
                    f'posterior_ridge_{suffix}_{param_name}.svg'
                    if suffix
                    else f'posterior_ridge_{param_name}.svg'
                ),
            )

        return fig, ax


def plot_group_violins(
    groups,
    ax=None,
    *,
    color=None,
    figsize=(8, 5),
    palette_range=(0.2, 0.8),
    y_label=None,
    label_fn=None,
    kde_width=0.1,
    viol_width=0.8,
    viol_lw=1,
    box_lw=5,
    whisker_lw=1.5,
    median_size=4,
    box_range=(25, 75),
    whisker_range=(0, 100),
    group_order=None,
    show_n=False,
    n_fontsize=None,
    savefig=False,
    save_name='group_violins.svg',
    fontsize=None,
    font_scale=1.0,
):
    """Violins of one variable for named groups ({name: values}), each with a
    whisker, a box and a median dot drawn at chosen percentiles. Group colours
    span `palette_range` of inferno unless `color` is given; `label_fn` maps
    group names to tick labels."""
    with plt.rc_context(font_rc(fontsize, font_scale)):
        frames = []
        for name, values in groups.items():
            vals = np.asarray(values).flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                frames.append(pd.DataFrame({'Value': vals, 'Group': name}))

        df = pd.concat(frames, ignore_index=True)
        names = list(groups.keys())
        if group_order is not None:
            names = [c for c in group_order if c in names] + [
                c for c in names if c not in group_order
            ]
        n = len(names)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_alpha(0)
        else:
            fig = ax.figure
        ax.patch.set_alpha(0)

        colors = plt.cm.inferno(np.linspace(palette_range[0], palette_range[1], n))
        if color:
            colors = [color] * n
        _pal_dict = {name: colors[i] for i, name in enumerate(names)}

        sns.violinplot(
            data=df,
            x='Group',
            y='Value',
            hue='Group',
            palette=_pal_dict,
            ax=ax,
            order=names,
            hue_order=names,
            linewidth=viol_lw,
            cut=0,
            bw_adjust=kde_width,
            width=viol_width,
            inner=None,
            density_norm='area',
            legend=False,
        )

        for i, name in enumerate(names):
            data = df[df['Group'] == name]['Value'].values
            p = np.percentile(
                data, [whisker_range[0], box_range[0], 50, box_range[1], whisker_range[1]]
            )
            ax.vlines(i, p[0], p[4], color='black', lw=whisker_lw, zorder=10)
            ax.vlines(i, p[1], p[3], color='black', lw=box_lw, zorder=11)
            ax.plot(i, p[2], 'wo', markersize=median_size, zorder=12, markeredgewidth=0)

        ax.set_xlabel('')
        ax.set_ylabel(y_label or '', labelpad=10)

        ymin, ymax = df['Value'].min(), df['Value'].max()
        modify_axis_spines(ax, which=['y'], yticks_main=[int(ymin), int(ymax)])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.set_xticklabels([label_fn(c) if label_fn else str(c) for c in names])

        if show_n:
            _nfs = n_fontsize or (plt.rcParams['font.size'] * 0.85)
            _ytop = ax.get_ylim()[1]
            for i, name in enumerate(names):
                _n_count = len(df[df['Group'] == name])
                ax.text(
                    i,
                    _ytop,
                    f'n={_n_count}',
                    ha='center',
                    va='bottom',
                    fontsize=_nfs,
                    color='black',
                    clip_on=False,
                )

        if fig is not None and len(fig.axes) == 1:
            plt.tight_layout()
        if savefig:
            save_figure(fig, save_name)

        return fig, ax


def plot_gradient_ridges(
    x,
    y,
    c,
    ax=None,
    *,
    xlim=None,
    x_round_to=None,
    cbar_ax='auto',
    n_bins=8,
    qcut=True,
    overlap=1.5,
    subset_quantiles=None,
    palette='inferno',
    alpha=0.85,
    color_norm='data',
    color_smooth_method='linear',
    kernel_bw_factor=5.0,
    ridge_lw=1.2,
    ridge_ec='black',
    baseline_lw=1.0,
    baseline_ls=':',
    baseline_ec='gray',
    baseline_alpha=0.3,
    figsize=(7, 6),
    x_precision=4,
    y_precision=2,
    c_precision=2,
    x_label=None,
    y_label=None,
    c_label=None,
    text_x_pos=0.98,
    text_y_offset=0.1,
    text_ha='right',
    label_sep=': ',
    text_fontweight='normal',
    savefig=False,
    save_name='gradient_ridges.svg',
    title=None,
    fontsize=None,
    font_scale=1.0,
):
    """Ridgelines of x for bins of y, each filled with a gradient that follows a
    third variable c along x.

    Within each bin c is smoothed against x (a linear fit, or a Gaussian kernel
    with `color_smooth_method='kernel'`) and painted under the bin's density.
    `color_norm` scales colours to the data ('data'), to what the ridges
    actually paint ('ridges'), or to explicit bounds. Each ridge is labelled
    with its bin's mean y.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        x = np.asarray(x)
        y = np.asarray(y)
        c_val = np.asarray(c)
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(c_val)
        x_clean, y_clean, c_clean = x[mask], y[mask], c_val[mask]
        if len(x_clean) == 0:
            raise ValueError("No valid overlapping data points found.")

        df = pd.DataFrame({'x': x_clean, 'y': y_clean, 'c': c_clean})
        if subset_quantiles is not None:
            df = df[
                (df['y'] >= df['y'].quantile(subset_quantiles[0]))
                & (df['y'] <= df['y'].quantile(subset_quantiles[1]))
            ]

        df['bin'] = (
            pd.qcut(df['y'], q=n_bins, duplicates='drop') if qcut else pd.cut(df['y'], bins=n_bins)
        )
        unique_bins = sorted(df['bin'].cat.categories, key=lambda b: b.left)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        c_min, c_max = df['c'].min(), df['c'].max()
        cmap = plt.get_cmap(palette)
        y_spacing = (y_max - y_min) / len(unique_bins) if len(unique_bins) > 1 else 1.0
        if y_spacing == 0:
            y_spacing = 1.0
        ridge_height = y_spacing * overlap
        y_ticks = []
        grid_x = np.linspace(x_min, x_max, 400)

        def _c_along(sub):
            """Colour variable as a smooth function of x within one ridge."""
            x_sub_min, x_sub_max = sub['x'].min(), sub['x'].max()
            if color_smooth_method == 'linear':
                res = stats.linregress(sub['x'], sub['c'])
                return np.clip(res.intercept + res.slope * grid_x, c_min, c_max)
            out = np.zeros_like(grid_x)
            bw = (x_sub_max - x_sub_min) / kernel_bw_factor if (x_sub_max - x_sub_min) > 0 else 1.0
            sub_bw = (x_sub_max - x_sub_min) * 0.15
            for j, gx in enumerate(grid_x):
                if gx < x_sub_min - sub_bw or gx > x_sub_max + sub_bw:
                    out[j] = np.nan
                    continue
                weights = np.exp(-0.5 * ((sub['x'] - gx) / bw) ** 2)
                w_sum = weights.sum()
                out[j] = np.average(sub['c'], weights=weights) if w_sum > 1e-6 else np.nan
            return out

        # Build first, draw second, so the colour scale can be set from what
        # actually gets rendered.
        _ridges = []
        for i, b in enumerate(unique_bins):
            sub = df[df['bin'] == b]
            if len(sub) < 3:
                continue
            try:
                kde = gaussian_kde(sub['x'])
            except Exception:
                continue
            pdf = kde(grid_x)
            pdf_norm = (pdf / pdf.max()) * ridge_height if pdf.max() > 0 else pdf
            c_smooth = _c_along(sub)
            valid_idx = np.where(~np.isnan(c_smooth) & (pdf_norm > 1e-4))[0]
            if len(valid_idx) < 2:
                continue
            _ridges.append((i, sub['y'].mean(), pdf_norm, c_smooth, valid_idx))
            y_ticks.append(sub['y'].mean())

        # 'ridges' normalises to the painted span, not the raw range of c:
        # a ridge's colour track covers only ~35% of the latter, so the
        # data-range default renders every ridge in the same mid-tones.
        if isinstance(color_norm, (tuple, list, np.ndarray)):
            _n_lo, _n_hi = float(color_norm[0]), float(color_norm[1])
        elif color_norm == 'ridges' and _ridges:
            _shown = np.concatenate([cs[vi[0] : vi[-1] + 1] for _, _, _, cs, vi in _ridges])
            _n_lo, _n_hi = float(np.nanmin(_shown)), float(np.nanmax(_shown))
        else:
            _n_lo, _n_hi = float(c_min), float(c_max)
        if not np.isfinite(_n_hi - _n_lo) or _n_hi <= _n_lo:
            _n_lo, _n_hi = float(c_min), float(c_max)
        norm = plt.Normalize(_n_lo, _n_hi)
        # round inward: a tick rounded just outside the norm gets no label
        _q = 10.0**c_precision
        _cbar_ticks = [np.ceil(_n_lo * _q) / _q, np.floor(_n_hi * _q) / _q]
        if _cbar_ticks[0] > _cbar_ticks[1]:
            _cbar_ticks = [round(_n_lo, c_precision), round(_n_hi, c_precision)]

        for i, baseline, pdf_norm, c_smooth, valid_idx in _ridges:
            first_valid, last_valid = valid_idx[0], valid_idx[-1]
            gx_valid = grid_x[first_valid : last_valid + 1]
            c_valid = c_smooth[first_valid : last_valid + 1]
            pdf_valid = pdf_norm[first_valid : last_valid + 1]
            pdf_max_val = pdf_valid.max()
            if pdf_max_val <= 0:
                continue
            gx_min_val, gx_max_val = gx_valid.min(), gx_valid.max()
            if gx_min_val == gx_max_val:
                continue

            extent = [gx_min_val, gx_max_val, baseline, baseline + pdf_max_val + 1e-5]
            im = ax.imshow(
                c_valid.reshape(1, -1),
                aspect='auto',
                cmap=cmap,
                norm=norm,
                extent=extent,
                origin='lower',
                alpha=alpha,
                zorder=i,
                interpolation='bicubic',
            )
            verts = (
                [(gx_min_val, baseline)]
                + list(zip(gx_valid, baseline + pdf_valid))
                + [(gx_max_val, baseline)]
            )
            poly = mpatches.Polygon(verts, facecolor='none', edgecolor='none')
            ax.add_patch(poly)
            im.set_clip_path(poly)
            if ridge_lw > 0:
                ax.plot(
                    gx_valid,
                    baseline + pdf_valid,
                    color=ridge_ec,
                    lw=ridge_lw,
                    alpha=0.8,
                    zorder=i + 0.1,
                )
            if baseline_lw > 0:
                ax.plot(
                    [x_min, x_max],
                    [baseline, baseline],
                    color=baseline_ec,
                    lw=baseline_lw,
                    ls=baseline_ls,
                    alpha=baseline_alpha,
                    zorder=i - 0.1,
                )
            label_text = f"{y_label}{label_sep}{baseline:.{y_precision}f}"
            ax.text(
                text_x_pos,
                baseline + ridge_height * text_y_offset,
                label_text,
                transform=ax.get_yaxis_transform(),
                ha=text_ha,
                va='bottom',
                fontweight=text_fontweight,
                color='k',
                zorder=i + 1,
            )

        ax.patch.set_alpha(0)
        ax.set_xlabel(x_label)
        if title:
            ax.set_title(title, fontweight='bold', pad=10)
        if xlim is not None:
            ax.set_xlim(*xlim)
        elif x_round_to:
            # Snap outward to a multiple of `x_round_to` so the axis reads as a
            # deliberate range rather than wherever the padded data happened to land.
            ax.set_xlim(
                np.floor(x_min / x_round_to) * x_round_to, np.ceil(x_max / x_round_to) * x_round_to
            )
        else:
            x_pad = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

        if len(y_ticks) > 0:
            actual_y_min, actual_y_max = min(y_ticks), max(y_ticks)
            ax.set_ylim(actual_y_min - (ridge_height * 0.05), actual_y_max + ridge_height * 1.05)
            modify_axis_spines(
                ax,
                which=['x'],
                xticks_main=[np.round(x_min, x_precision), np.round(x_max, x_precision)],
            )
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        if cbar_ax == 'auto':
            # only a figure of its own gets a colorbar; embedded panels leave
            # it to the caller
            if fig is not None and len(fig.axes) == 1:
                divider = make_axes_locatable(ax)
                _cax = divider.append_axes("right", size="3%", pad=0.2)
                add_styled_colorbar(sm, cax=_cax, label=c_label, ticks=_cbar_ticks)
        elif cbar_ax is not None:
            add_styled_colorbar(sm, cax=cbar_ax, label=c_label, ticks=_cbar_ticks)

        if savefig:
            save_figure(fig, save_name)

        return fig, ax


def plot_grouped_rainclouds(
    df,
    *,
    x,
    y,
    hue,
    order,
    hue_order,
    ax=None,
    palette=None,
    figsize=(12, 6),
    point_size=3,
    point_alpha=0.4,
    show_points=True,
    show_clouds=True,
    violin_width=0.65,
    violin_linewidth=0.0,
    kde_bw=0.4,
    cloud_alpha=0.7,
    box_width=0.15,
    box_linewidth=1.5,
    add_stats=True,
    test_type='Brunner-Munzel',
    comparisons_correction='BH',
    alpha=0.05,
    pvalues=None,
    stat_line_width=1.5,
    x_label=None,
    y_label=None,
    legend_title=None,
    legend_loc='upper right',
    legend_bbox_to_anchor=None,
    legend_ncols=1,
    ylim=None,
    title=None,
    savefig=False,
    save_name='grouped_rainclouds.svg',
    fontsize=None,
    font_scale=1.0,
):
    """Rainclouds of `y` for each category in `order`, split into two `hue`
    groups, with a significance bracket between the two halves of each category.

    Brackets come from statannotations running `test_type` (with
    `comparisons_correction`), or, when `pvalues` gives one final p-value per
    category, from those values with no further correction.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if palette is None:
            cmap = plt.get_cmap('inferno')
            palette_colors = [cmap(0.2), cmap(0.7)]
        elif isinstance(palette, str):
            cmap = plt.get_cmap(palette)
            palette_colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(hue_order))]
        elif isinstance(palette, dict):
            palette_colors = [palette[lbl] for lbl in hue_order]
        else:
            palette_colors = list(palette)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module='vizard')
            plot_rainclouds(
                x=x,
                y=y,
                hue=hue,
                data=df,
                ax=ax,
                palette=palette_colors,
                order=order,
                hue_order=hue_order,
                point_size=point_size if show_points else 0,
                alpha=point_alpha,
                dodge_boxes=True,
                plot_legend=False,
                box_alpha=0.8,
                clouds=show_clouds,
                boxes=True,
                rain=show_points,
                width_viol=violin_width,
                width_box=box_width,
                violin_linewidth=violin_linewidth,
                box_linewidth=box_linewidth,
                bw=kde_bw,
                cloud_alpha=cloud_alpha,
            )

        if add_stats:
            pairs = [((label, hue_order[0]), (label, hue_order[1])) for label in order]
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=df,
                    x=x,
                    y=y,
                    hue=hue,
                    order=order,
                    hue_order=hue_order,
                    width=box_width,
                    verbose=False,
                )
                annotator._pvalue_format = CustomPValueFormat()
                if pvalues is not None:
                    # p-values arrive final, so none of statannotations' own
                    # correction is applied on top
                    annotator.configure(
                        text_format='star', loc='inside', line_width=stat_line_width, alpha=alpha
                    )
                    annotator.set_pvalues_and_annotate(list(pvalues), num_comparisons=1)
                else:
                    annotator.configure(
                        test=test_type,
                        text_format='star',
                        loc='inside',
                        line_width=stat_line_width,
                        comparisons_correction=comparisons_correction,
                        alpha=alpha,
                    )
                    annotator.apply_test().annotate()
            except Exception as e:
                print(f"Stats annotation failed: {e}")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title, fontweight='bold', pad=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.patch.set_alpha(0)
        modify_axis_spines(ax, which=['y'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        legend_patches = [
            mpatches.Patch(color=palette_colors[i], label=hue_order[i])
            for i in range(len(hue_order))
        ]
        _leg_kw_rc = dict(
            handles=legend_patches,
            title=legend_title,
            frameon=False,
            loc=legend_loc,
            ncol=legend_ncols,
        )
        if legend_bbox_to_anchor is not None:
            _leg_kw_rc['bbox_to_anchor'] = legend_bbox_to_anchor
        ax.legend(**_leg_kw_rc)

        if ax is None or len(fig.axes) == 1:
            fig.tight_layout()
        if savefig:
            save_figure(fig, save_name)
        return fig, ax

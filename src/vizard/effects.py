"""Moderation effects: conditional slopes along a moderator or along age, and
a comparison of correlation strengths."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
from vizard.annotate import _pstars
from vizard.style import _auto_scale_exponent, font_rc, _nice_bounds, modify_axis_spines


def plot_conditional_slope_curve(
    ax,
    z_grid,
    slope_curve,
    ci_low,
    ci_high,
    decile_df=None,
    base_color='slategray',
    sig_color='indianred',
    ci_alpha=0.25,
    sig_ci_alpha=0.35,
    sig_bg_alpha=0.12,
    zero_line=True,
    ref_line=True,
    x_lim=None,
    x_label=None,
    y_label=None,
    title=None,
    fontsize=None,
    font_scale=1.0,
):
    """Slope of a predictor as a smooth function of a moderator, with its
    confidence band.

    The curve is drawn in `base_color` and switches to `sig_color`, over a
    faint tint, wherever the band excludes zero. `decile_df` (columns
    sc_resid_center, slope, ci_low, ci_high) adds binned slope estimates as
    points so the curve can be checked against them.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        z_grid = np.asarray(z_grid)
        ci_low = np.asarray(ci_low, dtype=float)
        ci_high = np.asarray(ci_high, dtype=float)
        slope_curve = np.asarray(slope_curve, dtype=float)

        exp = _auto_scale_exponent(np.concatenate([ci_low, ci_high, slope_curve]))
        scale = 10.0**exp
        ci_low_s, ci_high_s, slope_curve_s = ci_low / scale, ci_high / scale, slope_curve / scale

        sig = (ci_low > 0) | (ci_high < 0)
        edges = np.diff(sig.astype(int)) if sig.size else np.array([], dtype=int)
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if sig.size and sig[0]:
            starts = [0] + starts
        if sig.size and sig[-1]:
            ends = ends + [len(sig) - 1]
        sig_segments = [(s, min(e, len(z_grid) - 1)) for s, e in zip(starts, ends)]

        if zero_line:
            ax.axhline(0, color='black', lw=0.5, zorder=1)
        if ref_line:
            ax.axvline(0, color='dimgray', lw=0.5, ls=(0, (2, 2)), zorder=1)

        ax.fill_between(
            z_grid, ci_low_s, ci_high_s, color=base_color, alpha=ci_alpha, linewidth=0, zorder=2
        )
        ax.plot(z_grid, slope_curve_s, color=base_color, lw=1.8, zorder=3)

        for s, e in sig_segments:
            if e <= s:
                continue
            ax.axvspan(
                z_grid[s], z_grid[e], color=sig_color, alpha=sig_bg_alpha, zorder=0, linewidth=0
            )
            ax.fill_between(
                z_grid[s : e + 1],
                ci_low_s[s : e + 1],
                ci_high_s[s : e + 1],
                color=sig_color,
                alpha=sig_ci_alpha,
                linewidth=0,
                zorder=4,
            )
            ax.plot(z_grid[s : e + 1], slope_curve_s[s : e + 1], color=sig_color, lw=2.0, zorder=5)

        if decile_df is not None:
            valid = decile_df['slope'].notna()
            d = decile_df.loc[valid]
            yerr = np.vstack([d['slope'] - d['ci_low'], d['ci_high'] - d['slope']]) / scale
            ax.errorbar(
                d['sc_resid_center'],
                d['slope'] / scale,
                yerr=yerr,
                fmt='o',
                ms=3,
                color='black',
                ecolor='dimgray',
                elinewidth=0.8,
                capsize=0,
                zorder=6,
                alpha=0.85,
            )

        y_label = y_label or 'Conditional slope'
        if exp != 0:
            y_label = y_label + rf' ($\times 10^{{{exp}}}$)'
        ax.set_xlabel(x_label or 'Moderator (age-residual, z)', labelpad=2)
        ax.set_ylabel(y_label, labelpad=2)
        if title:
            ax.set_title(title, fontsize=fontsize, pad=10)

        # limits cover the binned points too, which may extend past the curve
        x_candidates = [z_grid]
        y_candidates = [ci_low_s, ci_high_s]
        if decile_df is not None and len(d):
            x_candidates.append(d['sc_resid_center'].to_numpy(dtype=float))
            y_candidates.append((d['ci_low'].to_numpy(dtype=float)) / scale)
            y_candidates.append((d['ci_high'].to_numpy(dtype=float)) / scale)
        x_all = np.concatenate(x_candidates)
        y_all = np.concatenate(y_candidates)
        x_lo, x_hi = _nice_bounds(float(np.nanmin(x_all)), float(np.nanmax(x_all)), decimals=1)
        y_lo, y_hi = _nice_bounds(float(np.nanmin(y_all)), float(np.nanmax(y_all)), decimals=1)
        if x_lim is not None:
            x_lo = x_lim[0] if x_lim[0] is not None else x_lo
            x_hi = x_lim[1] if x_lim[1] is not None else x_hi
        xticks = [x_lo, 0.0, x_hi] if x_lo < 0 < x_hi else [x_lo, x_hi]
        yticks = [y_lo, 0.0, y_hi] if y_lo < 0 < y_hi else [y_lo, y_hi]
        modify_axis_spines(
            ax, which=['x', 'y'], xticks_main=sorted(set(xticks)), yticks_main=sorted(set(yticks))
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.patch.set_alpha(0)
        return ax


def plot_lifespan_slope_dissociation(
    ax,
    curve_df,
    pooled=None,
    deg_color='indianred',
    pre_color='slategray',
    fill_deg=True,
    fill_pre=False,
    ci_alpha=0.18,
    ci_edge_lw=0.6,
    sig_lw=2.0,
    base_lw=1.1,
    base_alpha=0.45,
    base_ls=(0, (3, 2)),
    sig_marker_size=11,
    zero_line=True,
    show_legend=True,
    legend_loc='upper left',
    deg_label='SC-Deg',
    pre_label='SC-Pre',
    annotate_pooled=True,
    pooled_pos=(0.98, 0.04),
    x_label=None,
    y_label=None,
    title=None,
    x_lim=None,
    fontsize=None,
    font_scale=1.0,
):
    """Slopes in two subgroups along sliding windows, typically of age.

    `curve_df` holds the window centre and, for the deg_ and pre_ subgroups,
    slope, ci_low and ci_high. Windows whose interval excludes zero are drawn
    bold with a marker, the rest faint and dashed. Since overlapping windows
    are not independent tests, `pooled` (the pooled contrast, with 'slope' and
    'p' under 'diff') can be printed in the corner.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        d = curve_df[curve_df['deg_slope'].notna()]
        centers = d['center'].to_numpy(dtype=float)

        all_vals = np.concatenate(
            [d['deg_ci_low'], d['deg_ci_high'], d['pre_ci_low'], d['pre_ci_high']]
        )
        exp = _auto_scale_exponent(all_vals)
        scale = 10.0**exp

        if zero_line:
            ax.axhline(0, color='black', lw=0.5, zorder=1)

        # the reference goes first so the focal curve sits on top; filling both
        # bands would blend them into a third, unlabelled colour
        handles = {}
        for lo_c, hi_c, sl_c, color, lbl, fill in [
            ('pre_ci_low', 'pre_ci_high', 'pre_slope', pre_color, pre_label, fill_pre),
            ('deg_ci_low', 'deg_ci_high', 'deg_slope', deg_color, deg_label, fill_deg),
        ]:
            lo = d[lo_c].to_numpy(dtype=float) / scale
            hi = d[hi_c].to_numpy(dtype=float) / scale
            sl = d[sl_c].to_numpy(dtype=float) / scale
            if fill:
                ax.fill_between(centers, lo, hi, color=color, alpha=ci_alpha, linewidth=0, zorder=2)
            if ci_edge_lw:
                for edge in (lo, hi):
                    ax.plot(
                        centers,
                        edge,
                        color=color,
                        lw=ci_edge_lw,
                        alpha=0.55 if fill else 0.8,
                        ls='-' if fill else (0, (1, 1)),
                        zorder=2.5,
                    )
            ax.plot(centers, sl, color=color, lw=base_lw, alpha=base_alpha, ls=base_ls, zorder=3)
            handles[lbl] = mlines.Line2D([], [], color=color, lw=sig_lw, label=lbl)
            sig = (lo > 0) | (hi < 0)
            if sig.any():
                seg = np.where(sig, sl, np.nan)
                ax.plot(centers, seg, color=color, lw=sig_lw, zorder=4)
                ax.plot(
                    centers[sig],
                    sl[sig],
                    ls='none',
                    marker='o',
                    ms=np.sqrt(sig_marker_size),
                    color=color,
                    markeredgecolor='white',
                    markeredgewidth=0.6,
                    zorder=5,
                )

        y_lab = y_label or 'Conditional slope'
        if exp != 0:
            y_lab = y_lab + rf' ($\times 10^{{{exp}}}$)'
        ax.set_xlabel(x_label or 'Age (window centre)', labelpad=2)
        ax.set_ylabel(y_lab, labelpad=2)
        if title:
            ax.set_title(title, fontsize=fontsize, pad=10)

        y_all = all_vals / scale
        y_lo, y_hi = _nice_bounds(float(np.nanmin(y_all)), float(np.nanmax(y_all)), decimals=1)
        x_lo, x_hi = float(centers.min()), float(centers.max())
        if x_lim is not None:
            x_lo = x_lim[0] if x_lim[0] is not None else x_lo
            x_hi = x_lim[1] if x_lim[1] is not None else x_hi
        yticks = [y_lo, 0.0, y_hi] if y_lo < 0 < y_hi else [y_lo, y_hi]
        modify_axis_spines(
            ax,
            which=['x', 'y'],
            xticks_main=[x_lo, (x_lo + x_hi) / 2, x_hi],
            yticks_main=sorted(set(yticks)),
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        if annotate_pooled and pooled is not None:
            delta = pooled['diff']['slope'] / scale
            stars = _pstars(pooled['diff']['p'])
            txt = (
                rf"pooled $\Delta$ = {delta:.2f} "
                if abs(delta) >= 1
                else rf"pooled $\Delta$ = {delta:.3f} "
            ) + stars
            ax.annotate(
                txt,
                xy=pooled_pos,
                xycoords='axes fraction',
                ha='right',
                va='bottom',
                fontsize=fontsize,
            )

        if show_legend:
            order = [deg_label, pre_label]
            ax.legend(
                [handles[k] for k in order if k in handles],
                [k for k in order if k in handles],
                loc=legend_loc,
                frameon=False,
                handlelength=1.2,
                borderpad=0.2,
                labelspacing=0.25,
                handletextpad=0.5,
                fontsize=fontsize,
            )
        ax.patch.set_alpha(0)
        return ax


def plot_pred_score_dumbbell(
    target,
    pairs,
    pair_labels,
    ax=None,
    figsize=None,
    theta_color='indianred',
    feat_color='slategray',
    corr_kind='spearman',
    target_label='Age',
    dot_size=90,
    dot_alpha=0.92,
    line_lw=2.5,
    line_alpha=0.45,
    track_color='#e8e8e8',
    track_alpha=0.55,
    show_values=True,
    value_fontsize=None,
    x_lim='auto',
    row_spacing=1.0,
    legend_loc='upper center',
    legend_ncols=2,
    legend_bbox_to_anchor=None,
    theta_label='Latent',
    feat_label='Observed',
    labels_side='right',
    yticklabel_pad=None,
    x_label=None,
    fontsize=None,
    font_scale=1.0,
):
    """Absolute correlation with `target` of two competing predictors per row,
    drawn as two dots joined by a line in the stronger one's colour.

    `pairs` holds (latent values, observed values) per row and `pair_labels`
    their names. x_lim='auto' rounds the largest correlation up to the next 0.1.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        n_pairs = len(pairs)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.0, 0.85 * n_pairs + 0.9))

        t = np.asarray(target.values if hasattr(target, 'values') else target).ravel()
        fs = fontsize or 7
        val_fs = value_fontsize or fs * 0.85

        def _rho(a):
            m = ~np.isnan(a) & ~np.isnan(t)
            if m.sum() > 2:
                r = (
                    stats.spearmanr(t[m], a[m])[0]
                    if corr_kind == 'spearman'
                    else stats.pearsonr(t[m], a[m])[0]
                )
                return abs(r)
            return 0.0

        theta_scores, feat_scores = [], []
        for theta_vals, feat_vals in pairs:
            tv = np.asarray(
                theta_vals.values if hasattr(theta_vals, 'values') else theta_vals
            ).ravel()
            fv = np.asarray(feat_vals.values if hasattr(feat_vals, 'values') else feat_vals).ravel()
            theta_scores.append(_rho(tv))
            feat_scores.append(_rho(fv))

        if x_lim == 'auto':
            max_rho = max(theta_scores + feat_scores) if (theta_scores or feat_scores) else 1.0
            x_upper = (int(max_rho / 0.1) + 1) * 0.1
            x_lim = (0.0, round(x_upper, 10))

        y_positions = np.arange(n_pairs) * row_spacing

        for i, (ts, fs_val, (t_lbl, f_lbl)) in enumerate(
            zip(theta_scores, feat_scores, pair_labels)
        ):
            y = y_positions[i]
            lo, hi = min(ts, fs_val), max(ts, fs_val)

            ax.barh(
                y,
                x_lim[1] - x_lim[0],
                left=x_lim[0],
                height=0.38 * row_spacing,
                color=track_color,
                alpha=track_alpha,
                zorder=1,
            )

            winner_color = theta_color if ts >= fs_val else feat_color
            ax.plot(
                [lo, hi],
                [y, y],
                color=winner_color,
                lw=line_lw,
                alpha=line_alpha,
                solid_capstyle='round',
                zorder=2,
            )

            ax.scatter(
                fs_val,
                y,
                s=dot_size,
                color=feat_color,
                zorder=4,
                alpha=dot_alpha,
                edgecolors='white',
                linewidths=0.8,
            )
            ax.scatter(
                ts,
                y,
                s=dot_size,
                color=theta_color,
                zorder=5,
                alpha=dot_alpha,
                edgecolors='white',
                linewidths=0.8,
            )

            if show_values:
                # dot radius in points, from its area, plus a 2 pt gap
                text_pt = float(np.sqrt(dot_size / np.pi) + 2)
                for val, color, ha in [
                    (ts, theta_color, 'left' if ts > fs_val else 'right'),
                    (fs_val, feat_color, 'left' if fs_val > ts else 'right'),
                ]:
                    sign = 1 if ha == 'left' else -1
                    ax.annotate(
                        f'{val:.2f}',
                        xy=(val, y),
                        xytext=(sign * text_pt, 0),
                        textcoords='offset points',
                        va='center',
                        ha=ha,
                        fontsize=val_fs,
                        color=color,
                        fontweight='bold',
                    )

        row_tick_labels = [f'{tl} / {fl}' for tl, fl in pair_labels]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(row_tick_labels, fontsize=fs)
        if labels_side == 'right':
            ax.yaxis.tick_right()
        if yticklabel_pad is not None:
            ax.tick_params(axis='y', pad=yticklabel_pad)
        ax.set_xlim(x_lim)
        ax.set_ylim(-0.6 * row_spacing, (n_pairs - 0.4) * row_spacing)
        ax.set_xlabel(x_label or fr'$|\rho_S|$ with {target_label}', fontsize=fs)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        modify_axis_spines(ax, which=['x'], xticks_main=list(x_lim))
        ax.patch.set_alpha(0)

        legend_handles = [
            mpatches.Patch(color=theta_color, label=theta_label),
            mpatches.Patch(color=feat_color, label=feat_label),
        ]
        _leg_kw = dict(
            handles=legend_handles,
            frameon=False,
            loc=legend_loc,
            ncols=legend_ncols,
            fontsize=fs * 0.85,
            handlelength=1.0,
            handleheight=0.8,
        )
        if legend_bbox_to_anchor is not None:
            _leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
        ax.legend(**_leg_kw)

        return ax

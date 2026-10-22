"""Circular layouts of brain regions: chord diagrams of connectivity and
polar bars of regional values, sharing one node arrangement."""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path
from vizard.style import font_rc, add_styled_colorbar
from vizard.matrices import matrix_title_kw, matrix_title_pad


def _hemi_split_angles(n, node_hemi, gap, node_labels=None, singleton_boost=1.0, boost_mask=None):
    """Node angles with the left hemisphere on the left semicircle and the right
    on the right, `gap` radians apart at the poles.

    Nodes are evenly spaced unless `singleton_boost` > 1, which widens the slot
    of the nodes flagged in `boost_mask`. Without a mask, nodes whose label
    group (label minus trailing digits) has a single member are widened, which
    also catches groups that are singletons only by accident of the
    parcellation; pass the mask when the nodes that need room are known.
    """

    def _weighted_angles(idx, start, end):
        if len(idx) == 0:
            return np.array([])
        if singleton_boost == 1.0:
            return np.linspace(start, end, max(len(idx), 1))
        if boost_mask is not None:
            weights = np.array([singleton_boost if boost_mask[i] else 1.0 for i in idx])
        elif node_labels is not None:
            keys = [re.sub(r'\d+$', '', str(node_labels[i])) for i in idx]
            counts = {}
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            weights = np.array([singleton_boost if counts[k] == 1 else 1.0 for k in keys])
        else:
            return np.linspace(start, end, max(len(idx), 1))
        if len(idx) == 1:
            return np.array([(start + end) / 2.0])
        cum = np.cumsum(weights)
        mid_frac = (cum - weights / 2.0) / cum[-1]
        return start + mid_frac * (end - start)

    h = np.asarray(node_hemi)
    l_idx = np.where(h == 'L')[0]
    r_idx = np.where(h == 'R')[0]
    angles = np.empty(n)
    angles[l_idx] = _weighted_angles(l_idx, np.pi / 2 + gap / 2, 3 * np.pi / 2 - gap / 2)
    angles[r_idx] = _weighted_angles(r_idx, -np.pi / 2 + gap / 2, np.pi / 2 - gap / 2)
    return angles


def _group_node_spans(hemi_angles, keys):
    """Runs of equal keys along one hemisphere's ordered node angles, as
    (first angle, last angle, key, size)."""
    if len(keys) == 0:
        return []
    runs = []
    i0 = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[i0]:
            runs.append((hemi_angles[i0], hemi_angles[i - 1], keys[i0], i - i0))
            i0 = i
    return runs


def plot_radial_connectivity(
    matrices,
    ax=None,
    figsize=None,
    colors=None,
    labels=None,
    node_labels=None,
    label_fontsize=5.5,
    label_pad=1.08,
    label_mode='group',
    node_hemi=None,
    threshold='nonzero',
    threshold_value=None,
    alpha=0.5,
    lw_scale=1.0,
    lw_clip=(0.2, 3.0),
    node_radius=1.0,
    node_size=3,
    node_color='black',
    node_edgecolor='none',
    hemi_split=False,
    hemi_gap_deg=12,
    singleton_boost=1.0,
    boost_mask=None,
    curve_strength=0.6,
    show_legend=True,
    legend_kwargs=None,
    outer_scale=1.25,
    show_group_ring=False,
    group_ring_color='slategray',
    group_ring_alpha=0.9,
    group_line_width=0.5,
    title=None,
    fontsize=None,
    font_scale=1.0,
):
    """Chord diagram of one or more connectivity matrices over a ring of nodes.

    Each edge kept by `threshold` ('nonzero', 'abs' or 'value' against
    `threshold_value`) becomes a curved chord whose width follows |w|, pulled
    toward the centre by `curve_strength` (0 straight, 1 through the origin).
    `hemi_split` puts each hemisphere on its own semicircle; `show_group_ring`
    traces a line through the nodes of each label group. Labels are one per
    group by default (`label_mode='group'`) or one per node ('all').
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (3.5, 3.5))

        mats = [
            np.asarray(m, dtype=float)
            for m in (matrices if isinstance(matrices, (list, tuple)) else [matrices])
        ]
        n = mats[0].shape[0]
        for m in mats:
            if m.shape != (n, n):
                raise ValueError(f"all matrices must be ({n}, {n})")

        if colors is None:
            cycle = (
                plt.rcParams['axes.prop_cycle']
                .by_key()
                .get('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            )
            colors = [cycle[i % len(cycle)] for i in range(len(mats))]
        elif not isinstance(colors, (list, tuple)):
            colors = [colors] * len(mats)
        alphas = alpha if isinstance(alpha, (list, tuple)) else [alpha] * len(mats)

        gap = np.deg2rad(hemi_gap_deg)
        if hemi_split and n >= 2:
            if node_hemi is not None:
                angles = _hemi_split_angles(
                    n,
                    node_hemi,
                    gap,
                    node_labels=node_labels,
                    singleton_boost=singleton_boost,
                    boost_mask=boost_mask,
                )
            else:
                half = n // 2
                ang_l = np.linspace(np.pi / 2 + gap / 2, 3 * np.pi / 2 - gap / 2, half)
                ang_r = np.linspace(-np.pi / 2 + gap / 2, np.pi / 2 - gap / 2, n - half)
                angles = np.concatenate([ang_l, ang_r])
        else:
            angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)

        x_n = node_radius * np.cos(angles)
        y_n = node_radius * np.sin(angles)

        # each group's line runs through its own nodes, first to last; a
        # singleton gets a dot instead
        if show_group_ring and node_labels is not None and hemi_split and node_hemi is not None:
            h = np.asarray(node_hemi)
            # scatter sizes are areas in points^2: this dot is as wide as the line
            _singleton_dot_s = np.pi * (group_line_width / 2.0) ** 2
            for hemi_val in ('L', 'R'):
                idx = np.where(h == hemi_val)[0]
                if len(idx) == 0:
                    continue
                keys = [re.sub(r'\d+$', '', str(node_labels[i])) for i in idx]
                hemi_angles = angles[idx]
                for t0, t1, _key, size in _group_node_spans(hemi_angles, keys):
                    if size == 1:
                        ax.scatter(
                            [node_radius * np.cos(t0)],
                            [node_radius * np.sin(t0)],
                            s=_singleton_dot_s,
                            c=group_ring_color,
                            alpha=group_ring_alpha,
                            edgecolors='none',
                            zorder=7,
                        )
                    else:
                        arc_t = np.linspace(t0, t1, max(int(np.degrees(t1 - t0) * 2), 2))
                        ax.plot(
                            node_radius * np.cos(arc_t),
                            node_radius * np.sin(arc_t),
                            color=group_ring_color,
                            lw=group_line_width,
                            alpha=group_ring_alpha,
                            solid_capstyle='round',
                            zorder=6,
                        )

        # chords are undirected, but a directed matrix may store an edge below
        # the diagonal only, so each pair takes the larger of its two entries
        rows, cols = np.triu_indices(n, k=1)
        legend_handles = []

        for m, c, a, lbl in zip(mats, colors, alphas, labels or [None] * len(mats)):
            w_upper = m[rows, cols]
            w_lower = m[cols, rows]
            w = np.where(np.abs(w_upper) >= np.abs(w_lower), w_upper, w_lower)
            absw = np.abs(w)
            if threshold == 'nonzero':
                keep = absw > 0
            elif threshold == 'abs':
                keep = absw >= (threshold_value or 0.0)
            else:
                keep = w >= (threshold_value or 0.0)
            if not keep.any():
                continue
            w_max = absw[keep].max() if absw[keep].max() > 0 else 1.0

            for r_idx, c_idx, ww in zip(rows[keep], cols[keep], w[keep]):
                x0, y0 = x_n[r_idx], y_n[r_idx]
                x1, y1 = x_n[c_idx], y_n[c_idx]
                xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                xc, yc = xm * (1.0 - curve_strength), ym * (1.0 - curve_strength)
                verts = [(x0, y0), (xc, yc), (x1, y1)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                lw = np.clip(lw_scale * (abs(ww) / w_max), lw_clip[0], lw_clip[1])
                ax.add_patch(
                    mpatches.PathPatch(
                        Path(verts, codes), edgecolor=c, lw=lw, facecolor='none', alpha=a
                    )
                )
            if lbl is not None:
                legend_handles.append(mpatches.Patch(color=c, label=lbl))

        ax.scatter(x_n, y_n, s=node_size, c=node_color, edgecolors=node_edgecolor, zorder=5)

        if node_labels is not None:
            if label_mode == 'group':
                _groups = {}
                for angle, lbl in zip(angles, node_labels):
                    key = re.sub(r'\d+$', '', str(lbl))
                    _groups.setdefault(key, []).append(angle)
                _label_items = []
                for key, angs in _groups.items():
                    mid = np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs)))
                    _label_items.append((mid, key))
            else:
                _label_items = list(zip(angles, node_labels))
            for angle, lbl in _label_items:
                xl = label_pad * node_radius * np.cos(angle)
                yl = label_pad * node_radius * np.sin(angle)
                rot = np.degrees(angle)
                ha = 'left' if np.cos(angle) >= 0 else 'right'
                if np.cos(angle) < 0:
                    rot += 180
                t = ax.text(
                    xl,
                    yl,
                    lbl,
                    ha=ha,
                    va='center',
                    fontsize=label_fontsize,
                    rotation=rot,
                    rotation_mode='anchor',
                    clip_on=False,
                )
                t.set_clip_path(None)
        else:
            _label_items = []

        ax.set_aspect('equal')
        ax.set_clip_on(False)
        # labels need outer_scale >= 1.2, or the top ones press against the
        # axes edge and look closer to the ring than the bottom ones
        _eff_outer = max(outer_scale, 1.2) if node_labels is not None else outer_scale
        lim = (
            (node_radius * label_pad * _eff_outer)
            if node_labels is not None
            else node_radius * _eff_outer * 0.88
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_alpha(0)
        if title:
            ax.set_title(title, pad=matrix_title_pad, **matrix_title_kw)

        if show_legend and legend_handles:
            kw = {'frameon': False, 'loc': 'center', 'bbox_to_anchor': (0.5, -0.02)}
            kw.update(legend_kwargs or {})
            ax.legend(handles=legend_handles, **kw)

        return ax


def plot_radial_values(
    values,
    ax,
    node_colors=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    hemi_split=False,
    hemi_gap_deg=12,
    singleton_boost=1.0,
    boost_mask=None,
    bar_alpha=1.0,
    bar_lw=0.5,
    bar_edgecolor='black',
    node_labels=None,
    label_fontsize=5.5,
    label_mode='group',
    node_hemi=None,
    node_size=8,
    node_color='black',
    node_edgecolor='none',
    bar_width_scale=0.9,
    outer_scale=None,
    show_group_ring=False,
    group_ring_color='slategray',
    group_ring_alpha=0.9,
    group_line_width=0.5,
    show_cbar=False,
    cbar_label='Value',
    cbar_labelpad=4,
    fontsize=None,
    font_scale=1.0,
):
    """Polar bars of one value per node, laid out exactly as
    plot_radial_connectivity so the two can sit side by side.

    `ax` must be polar, e.g. ``fig.add_subplot(gs[0, 1], projection='polar')``.
    """
    with plt.rc_context(font_rc(fontsize, font_scale)):
        v = np.asarray(values, dtype=float)
        n = len(v)
        half = n // 2

        gap = np.deg2rad(hemi_gap_deg)
        if hemi_split and n >= 2:
            if node_hemi is not None:
                theta = _hemi_split_angles(
                    n,
                    node_hemi,
                    gap,
                    node_labels=node_labels,
                    singleton_boost=singleton_boost,
                    boost_mask=boost_mask,
                )
            else:
                ang_l = np.linspace(np.pi / 2 + gap / 2, 3 * np.pi / 2 - gap / 2, half)
                ang_r = np.linspace(-np.pi / 2 + gap / 2, np.pi / 2 - gap / 2, n - half)
                theta = np.concatenate([ang_l, ang_r])
        else:
            theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)

        # boosted layouts are uneven, so each bar takes its local node spacing
        if singleton_boost != 1.0 and hemi_split and node_hemi is not None:
            order = np.argsort(theta)
            sorted_theta = theta[order]
            gaps = np.diff(sorted_theta)
            local = np.empty(n)
            local[order[0]] = gaps[0] if len(gaps) else (2 * np.pi - 2 * gap) / n
            local[order[-1]] = gaps[-1] if len(gaps) else (2 * np.pi - 2 * gap) / n
            if len(gaps) > 1:
                local[order[1:-1]] = (gaps[:-1] + gaps[1:]) / 2.0
            bar_width = local * bar_width_scale
        else:
            bar_width = (2 * np.pi - 2 * gap) / n if hemi_split else 2 * np.pi / n
            bar_width *= bar_width_scale

        if node_colors is None:
            lo = vmin if vmin is not None else np.nanmin(v)
            hi = vmax if vmax is not None else np.nanmax(v)
            norm = mcolors.Normalize(vmin=lo, vmax=hi)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cols = sm.to_rgba(v)
        else:
            cols = node_colors
            sm = None

        # bar() draws no outline unless edgecolor is set, whatever the linewidth
        ax.bar(
            theta,
            v,
            width=bar_width,
            color=cols,
            alpha=bar_alpha,
            linewidth=bar_lw,
            edgecolor=bar_edgecolor,
            bottom=0.0,
        )

        # the largest bar nearly reaches the node ring, and the radial limit
        # matches the chord diagram so both rings come out the same size
        _hi = vmax if vmax is not None else np.nanmax(v)
        r_ring = _hi / 0.97
        _r_lbl = r_ring * 1.03
        _os = outer_scale if outer_scale is not None else 1.2875
        ax.set_rlim(0, r_ring * _os)

        ax.scatter(
            theta,
            np.full(n, r_ring),
            s=node_size,
            c=node_color,
            edgecolors=node_edgecolor,
            zorder=5,
        )

        if show_group_ring and node_labels is not None and hemi_split and node_hemi is not None:
            h = np.asarray(node_hemi)
            _singleton_dot_s = np.pi * (group_line_width / 2.0) ** 2
            for hemi_val in ('L', 'R'):
                idx = np.where(h == hemi_val)[0]
                if len(idx) == 0:
                    continue
                keys = [re.sub(r'\d+$', '', str(node_labels[i])) for i in idx]
                hemi_theta = theta[idx]
                for t0, t1, _key, size in _group_node_spans(hemi_theta, keys):
                    if size == 1:
                        ax.scatter(
                            [t0],
                            [r_ring],
                            s=_singleton_dot_s,
                            c=group_ring_color,
                            alpha=group_ring_alpha,
                            edgecolors='none',
                            zorder=7,
                        )
                    else:
                        arc_t = np.linspace(t0, t1, max(int(np.degrees(t1 - t0) * 2), 2))
                        ax.plot(
                            arc_t,
                            np.full_like(arc_t, r_ring),
                            color=group_ring_color,
                            lw=group_line_width,
                            alpha=group_ring_alpha,
                            solid_capstyle='round',
                            zorder=6,
                        )

        ax.set_xticks([])
        if node_labels is not None:
            if label_mode == 'group':
                _grp = {}
                for t, lbl in zip(theta, node_labels):
                    key = re.sub(r'\d+$', '', str(lbl))
                    _grp.setdefault(key, []).append(t)
                _gt = [np.arctan2(np.mean(np.sin(ts)), np.mean(np.cos(ts))) for ts in _grp.values()]
                _gl = list(_grp.keys())
            else:
                _gt, _gl = list(theta), list(node_labels)
            for t_ang, lbl in zip(_gt, _gl):
                rot = np.degrees(t_ang)
                ha = 'left' if np.cos(t_ang) >= 0 else 'right'
                if np.cos(t_ang) < 0:
                    rot += 180
                ax.text(
                    t_ang,
                    _r_lbl,
                    lbl,
                    ha=ha,
                    va='center',
                    fontsize=label_fontsize,
                    rotation=rot,
                    rotation_mode='anchor',
                    clip_on=False,
                )

        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        ax.patch.set_alpha(0)
        ax.set_clip_on(False)

        # an inset colorbar leaves the axes box, and so the alignment with a
        # neighbouring chord diagram, untouched
        if show_cbar and sm is not None:
            cax = ax.inset_axes([1.05, 0.1, 0.06, 0.8])
            add_styled_colorbar(
                sm,
                cax=cax,
                label=cbar_label,
                ticks=[lo, hi],
                labelpad=cbar_labelpad,
                orientation='vertical',
                fontsize=fontsize,
                font_scale=font_scale,
            )

        return ax

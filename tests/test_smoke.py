"""Every public plotting function draws on small synthetic data without error."""

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pytest

import vizard as vz


@pytest.fixture(autouse=True)
def _close_figures():
    np.random.seed(0)
    yield
    plt.close('all')


rng = np.random.default_rng(0)


def _sym(n):
    m = rng.random((n, n))
    return (m + m.T) / 2


def test_formatters():
    assert vz.auto_ticks([-0.003, 0.002], 2) == [-0.003, 0.002]
    assert vz.latex_sci_fmt(0.0001) == r'$1\times10^{-4}$'
    assert vz.compact_count_label(5000) == '5e3'
    assert vz.font_rc(9)['axes.labelsize'] == 9.0


def test_save_figure(tmp_path):
    vz.set_figure_dir(str(tmp_path))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    vz.save_figure(fig, 'line.svg')
    assert (tmp_path / 'line.svg').exists()


def test_layout():
    fig, axd = plt.subplot_mosaic('AB')
    vz.annotate_panels(axd, {'A': 'a', 'B': 'b'})
    fs = vz.two_pass_calibrated_fontsize(lambda f: plt.subplots(figsize=(3, 2)), 3)
    assert fs > 0


def test_rainclouds():
    df = pd.DataFrame(
        {'g': np.repeat(['a', 'b'], 30), 'h': np.tile(['lo', 'hi'], 30), 'v': rng.normal(size=60)}
    )
    fig, ax = plt.subplots()
    vz.plot_rainclouds(x='g', y='v', data=df, ax=ax, order=['a', 'b'])
    vz.plot_grouped_rainclouds(df, x='g', y='v', hue='h', order=['a', 'b'], hue_order=['lo', 'hi'])
    vz.plot_grouped_rainclouds(
        df, x='g', y='v', hue='h', order=['a', 'b'], hue_order=['lo', 'hi'], pvalues=[0.01, 0.5]
    )


def test_violins_and_ridges():
    vz.plot_group_violins({'a': rng.normal(50, 5, 40), 'b': rng.normal(60, 5, 40)}, show_n=True)
    post = xr.DataArray(
        rng.normal(size=(40, 50, 1)), dims=('subject', 'sample', 'theta'), coords={'theta': ['tau']}
    )
    vz.plot_posterior_ridge(post, 'tau', rng.uniform(20, 80, 40), bin_size=20, plot_trend=True)
    x, y, c = rng.normal(size=200), rng.uniform(20, 80, 200), rng.normal(size=200)
    vz.plot_gradient_ridges(x, y, c, n_bins=4, x_label='x', y_label='y', c_label='c')


def test_regressions():
    x, y = rng.normal(size=100), rng.normal(size=100)
    fig, ax = plt.subplots()
    vz.plot_scatter_with_regression(ax, x, y, c=rng.random(100), cmap='viridis', subsample=0.5)
    vz.plot_regression_layer(ax, x, y, 'k', 'r', 3, 0.5, 1)
    fig, ax = plt.subplots()
    vz.plot_agreement_hexbin(np.abs(x), np.abs(y), ax, gridsize=10)
    df = pd.DataFrame(
        {
            'age': rng.uniform(20, 80, 120),
            'v': rng.normal(size=120),
            'w': rng.normal(size=120),
            'grp': np.repeat(['a', 'b', 'c'], 40),
        }
    )
    vz.plot_group_regressions(df, x_col='age', y_col='v', group_col='grp')
    vz.plot_group_regression_grid(df, ['v', 'w'], x_var='age', group_col='grp')
    vz.plot_binned_overlay(df['age'], df['v'], df['w'], show_cbar=True)


def test_matrices():
    vz.plot_sc_template(_sym(20))
    vz.plot_fc_template(_sym(20))
    vz.plot_fcd_template(_sym(15))
    vz.plot_distance_matrix(_sym(10))
    fig, ax = plt.subplots()
    vz.plot_sample_bold(rng.normal(size=(100, 5)), fig=fig, ax=ax)


def test_radial():
    n = 20
    hemi = ['L'] * 10 + ['R'] * 10
    labels = [f'{h}.Net{i % 3}' for h, i in zip(hemi, range(n))]
    vz.plot_radial_connectivity(
        _sym(n) > 0.8, hemi_split=True, node_hemi=hemi, node_labels=labels, show_group_ring=True
    )
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    vz.plot_radial_values(
        rng.random(n), ax, hemi_split=True, node_hemi=hemi, node_labels=labels, show_cbar=True
    )


def test_effects():
    z = np.linspace(-2, 2, 50)
    fig, ax = plt.subplots()
    vz.plot_conditional_slope_curve(ax, z, z * 0.1, z * 0.1 - 0.05, z * 0.1 + 0.05)
    centers = np.linspace(40, 80, 10)
    curve = pd.DataFrame(
        {
            'center': centers,
            'deg_slope': centers * 0.01,
            'deg_ci_low': centers * 0.01 - 0.1,
            'deg_ci_high': centers * 0.01 + 0.1,
            'pre_slope': np.zeros(10),
            'pre_ci_low': -np.ones(10) * 0.1,
            'pre_ci_high': np.ones(10) * 0.1,
        }
    )
    fig, ax = plt.subplots()
    vz.plot_lifespan_slope_dissociation(ax, curve, pooled={'diff': {'slope': 0.2, 'p': 0.01}})
    t = rng.normal(size=50)
    vz.plot_pred_score_dumbbell(t, [(t + rng.normal(size=50), rng.normal(size=50))], [('a', 'b')])


def test_sweeps_and_traces():
    sweep = np.linspace(0, 1, 6)
    n_subj = 3
    feats = np.stack([np.exp(-sweep * (i + 1)) for i in range(n_subj)]).reshape(-1, 1)
    feats = np.hstack([feats, feats**2])
    subj_idx = np.repeat(np.arange(n_subj), len(sweep))
    vz.plot_sweep_curves(
        feats,
        ['f1', 'f2'],
        sweep,
        subj_idx,
        np.arange(n_subj) * 1.0,
        show_cbar=True,
        legend_title='c',
    )
    rows, cols = np.linspace(0, 1, 4), np.linspace(0, 2, 5)
    vz.plot_sweep_heatmaps(
        rng.random((20, 2)), ['f1', 'f2'], rows, cols, x_tick_fmt=vz.latex_sci_fmt
    )
    fig, ax = plt.subplots()
    it = np.arange(1, 101)
    vz.plot_train_test_curves(
        ax, it, [{'y': 1 / it, 'label': 'train'}, {'y': 1.1 / it, 'label': 'test'}], log_y=True
    )

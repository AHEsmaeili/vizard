# vizard

Matplotlib wrappers for scientific figures, based on my previous publications. Inputs are arrays, data frames or xarray objects.

## Installation

```bash
pip install git+https://github.com/AHEsmaeili/vizard
```

## What is in it

Everything below can be imported from `vizard` directly.

| Module                 | Contents                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vizard.style`         | the house style: one font size shared across a figure, minimal spines with ticks only at chosen values, tick formatters, colorbars, and `save_figure` |
| `vizard.layout`        | panel letters, and font sizes calibrated to a figure's printed width                                                                                  |
| `vizard.distributions` | raincloud plots, grouped rainclouds with significance brackets, violins, ridgelines of posterior samples across bins, and gradient-filled ridgelines  |
| `vizard.regression`    | scatter plots with a fitted line and correlation, one panel per group, group-by-variable grids, binned overlays, and agreement hexbins                |
| `vizard.matrices`      | structural, functional and FCD matrices, pairwise distance matrices, and BOLD time series                                                             |
| `vizard.radial`        | connectivity and regional values on a circular layout split by hemisphere                                                                             |
| `vizard.effects`       | conditional-slope curves along a moderator or along age, and dumbbell plots comparing correlation strengths                                           |
| `vizard.sweeps`        | response curves over one swept parameter, and feature maps over a grid of two                                                                         |
| `vizard.traces`        | train and test loss curves                                                                                                                            |
| `vizard.annotate`      | p-value formatting for statannotations, and significance stars                                                                                        |

## Conventions

Every plotting function takes `fontsize` and `font_scale`. All text drawn by a
call (ticks, labels, legends, colorbars) uses the same size, which defaults to
`vizard.style.base_fontsize` (7 pt). Functions that draw on an existing axis
take `ax=`. Functions that build their own figure return `(fig, axes)`.

`save_figure(fig, name)` writes into the folder set with `set_figure_dir`.
Otherwise it writes to the working directory. It saves with a transparent
background and a tight bounding box.

Raincloud jitter is drawn from NumPy's global random state. Call
`numpy.random.seed` first if a figure must come out identical on every run.

## License

BSD-3-Clause. See `LICENSE`.

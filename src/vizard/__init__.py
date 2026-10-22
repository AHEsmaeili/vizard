"""Publication figure helpers built on matplotlib and seaborn.

Every plotting function takes an optional `ax`, draws into it, and sizes its
text from `fontsize` and `font_scale`, so panels compose into multi-panel
figures without restyling.
"""

from vizard.style import (
    set_figure_dir,
    save_figure,
    set_mpl_colors,
    font_rc,
    modify_axis_spines,
    auto_ticks,
    latex_sci_fmt,
    add_styled_colorbar,
    clean_colorbar,
    compact_count_label,
    add_custom_scalebar,
    plot_image_schema,
)
from vizard.annotate import CustomPValueFormat
from vizard.layout import annotate_panels, calibrated_fontsize, two_pass_calibrated_fontsize
from vizard.distributions import (
    plot_rainclouds,
    plot_grouped_rainclouds,
    plot_group_violins,
    plot_posterior_ridge,
    plot_gradient_ridges,
)
from vizard.regression import (
    plot_regression_layer,
    plot_scatter_with_regression,
    plot_group_regressions,
    plot_group_regression_grid,
    plot_binned_overlay,
    plot_agreement_hexbin,
)
from vizard.matrices import (
    plot_sc_template,
    plot_fc_template,
    plot_fcd_template,
    plot_sample_bold,
    plot_distance_matrix,
)
from vizard.radial import plot_radial_connectivity, plot_radial_values
from vizard.effects import (
    plot_conditional_slope_curve,
    plot_lifespan_slope_dissociation,
    plot_pred_score_dumbbell,
)
from vizard.sweeps import plot_sweep_curves, plot_sweep_heatmaps
from vizard.traces import plot_train_test_curves

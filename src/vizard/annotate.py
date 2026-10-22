"""Significance labels for statannotations brackets."""

import numpy as np

try:
    from statannotations.PValueFormat import PValueFormat
except ImportError:
    from statannotations.stats.PValueFormat import PValueFormat


class CustomPValueFormat(PValueFormat):
    """Brackets read 'p = 0.001*' when significant at the configured alpha,
    after any correction, and 'ns' otherwise."""

    def format_data(self, result):
        if not result.is_significant:
            return 'ns'
        width = ".3f" if result.pvalue >= 0.001 else ".1e"
        return f"p = {result.pvalue:{width}}{result.significance_suffix}"


def _pstars(p):
    """Star string for a p-value: * below 0.05 down to **** below 1e-4."""
    if p is None or not np.isfinite(p):
        return ''
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'

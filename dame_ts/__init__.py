"""
DAME-TS: Differentially Private Mean Estimation via Ternary Search
"""

from .dame_ts import dame_with_ternary_search
from .ternary_search import attempting_insertion_using_ternary_search
from .utils import theoretical_upper_bound, run_dame_experiment,plot_errorbars_and_upper_bounds

__all__ = [
    "dame_with_ternary_search",
    "plot_errorbars_and_upper_bounds",
    "theoretical_upper_bound",
    "run_dame_experiment",
    "attempting_insertion_using_ternary_search"
]

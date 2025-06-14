import pytest
from dame_ts.utils import min_n_required, theoretical_upper_bound,run_dame_experiment
import numpy as np




def test_min_n_required_basic():
    alpha = 0.6
    n = min_n_required(alpha)
    assert n > 0
    assert isinstance(n, float)

def test_min_n_required_edge_case():
    # Test very small alpha where (2pi_alpha - 1)^4 ~ 0
    alpha = 0.000001
    n = min_n_required(alpha)
    assert n == float("inf")

def test_theoretical_upper_bound_valid():
    alpha = 0.6
    n = 5000
    m = 20
    bound = theoretical_upper_bound(alpha, n, m)
    assert bound >= 0
    assert isinstance(bound, float)


def test_run_dame_experiment_normal():
    mean_err, std_err = run_dame_experiment(n=5000,
                     alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="normal")
    assert mean_err >= 0
    assert std_err >= 0

def test_run_dame_experiment_uniform():
    mean_err, std_err = run_dame_experiment(n=5000,
                     alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="uniform")
    assert mean_err >= 0
    assert std_err >= 0

def test_run_dame_experiment_laplace():
    mean_err, std_err = run_dame_experiment(n=5000,
                     alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="laplace")
    assert mean_err >= 0
    assert std_err >= 0

def test_run_dame_experiment_exponential():
    mean_err, std_err = run_dame_experiment(n=5000,
                     alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="exponential")
    assert mean_err >= 0
    assert std_err >= 0

import pytest
from dame_ts.utils import min_n_required, theoretical_upper_bound,run_dame_experiment
import numpy as np
from dame_ts.dame_ts import dame_with_ternary_search




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



def test_estimate_within_error_and_bound(tol=0.1):
    np.random.seed(42)

    true_mean = 0.3
    n = 4000
    m = 20
    alpha = 0.6
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]

    estimated_mean = dame_with_ternary_search(n, alpha, m, user_samples)
    error = abs(estimated_mean - true_mean)

    # Static small error tolerance
    # assert error < tol, f"Error {error:.4f} too large"

    # Theoretical bound check
    theoretical_bound = theoretical_upper_bound(alpha, n, m)  
    assert error <= theoretical_bound, f"Error {error:.4f} exceeds theoretical bound {theoretical_bound:.4f}"

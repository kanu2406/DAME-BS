import pytest
from dame_ts.utils import min_n_required, theoretical_upper_bound,run_dame_experiment,plot_errorbars_and_upper_bounds
import numpy as np
from dame_ts.dame_ts import dame_with_ternary_search
import matplotlib.pyplot as plt

########################################################################################

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

#####################################################################################
# Tests for different distributions

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


###########################################################################################
def test_estimate_within_error_and_bound(tol=0.1):
    np.random.seed(42)

    true_mean = 0.3
    n = 5000
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

##################################################################################################

def test_plot_errorbars_and_upper_bounds_runs(monkeypatch):
    # Monkeypatch plt.show to avoid displaying the plot during test
    monkeypatch.setattr(plt, "show", lambda: None)

    true_mean = 0.3
    n = 5000
    m = 20
    alpha = 0.6
    mean_errors = []
    std_errors = []
    alphas=[]
    distribution="normal"

    for alpha in alphas:
        
        alphas.append(alpha)
        mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials=50,distribution="normal")
        mean_errors.append(mean_err)
        std_errors.append(std_err)

    upper_bounds = [theoretical_upper_bound(a, n, m) for a in alphas]

    plot_errorbars_and_upper_bounds(alphas, mean_errors, std_errors,upper_bounds, 
                   xlabel="Privacy parameter α"
                   , ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Alpha for the {distribution} distribution")


    
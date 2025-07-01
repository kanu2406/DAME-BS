import pytest
from dame_bs.utils import theoretical_upper_bound,run_dame_experiment,plot_errorbars_and_upper_bounds
import numpy as np
from dame_bs.dame_bs import dame_with_binary_search
import matplotlib.pyplot as plt

########################################################################################

def test_theoretical_upper_bound_valid():
    '''Verifies that theoretical_upper_bound returns a non-negative float.'''
    alpha = 0.6
    n = 9000
    m = 20
    bound = theoretical_upper_bound(alpha, n, m,delta=0.1)
    assert bound >= 0
    assert isinstance(bound, float)

#####################################################################################
# Tests for different distributions

def test_run_dame_experiment_for_distributions():
    '''Runs the experiment using each supported distribution and asserts that errors are non-negative.'''

    distributions = ["normal","exponential","uniform","poisson"]
    for dist in distributions:
            mean_err, std_err = run_dame_experiment(n=9000,
                            alpha=0.6, m=20, true_mean=0.3, trials=50,distribution=dist,delta=0.1)
            assert mean_err >= 0
            assert std_err >= 0


###########################################################################################
def test_estimate_within_error_and_bound(tol=0.1):
    '''Ensures that the DAME-BS empirical error is less than theoretical bound.'''

    np.random.seed(42)

    true_mean = 0.3
    n = 9000
    m = 20
    alpha = 0.6
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]

    mean_err, std_err = run_dame_experiment(n=9000,
                            alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="normal",delta=0.1)

    # Theoretical bound check
    theoretical_bound = theoretical_upper_bound(alpha, n, m,delta=0.1)  
    assert mean_err <= theoretical_bound, f"Error {mean_err:.4f} exceeds theoretical bound {theoretical_bound:.4f}"

##################################################################################################

def test_plot_errorbars_and_upper_bounds_runs(monkeypatch):
    '''Ensures the plotting function executes without crashing. 
    Uses monkeypatch to suppress actual plot display.'''

    # Monkeypatch plt.show to avoid displaying the plot during test
    monkeypatch.setattr(plt, "show", lambda: None)

    true_mean = 0.3
    n = 9000
    m = 20
    alpha = 0.6
    delta=0.1
    mean_errors = []
    std_errors = []
    alphas=[]
    distribution="normal"

    for alpha in alphas:
        
        alphas.append(alpha)
        mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials=50,distribution="normal",delta=0.1)
        mean_errors.append(mean_err)
        std_errors.append(std_err)

    upper_bounds = [theoretical_upper_bound(a, n, m,delta) for a in alphas]

    plot_errorbars_and_upper_bounds(alphas, mean_errors, std_errors,upper_bounds, 
                   xlabel="Privacy parameter α"
                   , ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Alpha for the {distribution} distribution")

#############################################################################################

def test_theoretical_bound_no_noise():
    '''Tests the edge case where alpha = ∞ (i.e., no added noise), 
    and ensures the theoretical bound is reduced to its minimal form.'''

    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    pi_alpha = 1
    n=9000
    m = 20
    true_mean = 0.3
    delta=0.1
    
    mean_err,std_err = run_dame_experiment(n, alpha, m, true_mean, trials=50,distribution="normal",delta=0.1)
    theoretical_bound=theoretical_upper_bound(alpha, n, m,delta=0.1)
    assert mean_err < theoretical_bound, f" Mean squared error size is greater than theoretical upper bound"
    assert mean_err < 1e-4, f"Error is unexpectedly high without privacy: {mean_err:.6f}"
    assert theoretical_bound == (32/n) + (4*delta),f"Theoretical Upper bound still contains error due to noise"

#############################################################################################

def test_error_decreases_with_more_users():
    '''Ensures that emperical error reduces with more users'''
    err_few_users,_ = run_dame_experiment(n=1000, alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="normal",delta=0.1)
    err_many_users,_ = run_dame_experiment(n=10000, alpha=0.6, m=20, true_mean=0.3, trials=50,distribution="normal",delta=0.1)
    assert err_many_users < err_few_users, f"Error did not decrease with more users: {err_few_users:.4f} vs {err_many_users:.4f}"

#############################################################################################

def test_error_decreases_with_higher_alpha():
    '''Ensures that emperical error reduces with higher alpha (lesser privacy)'''
    err_low_alpha,_ = run_dame_experiment(n=10000, alpha=0.2, m=20, true_mean=0.3, trials=50,distribution="normal",delta=0.1)
    err_high_alpha,_ = run_dame_experiment(n=10000, alpha=0.8, m=20, true_mean=0.3, trials=50,distribution="normal",delta=0.1)
    assert err_high_alpha < err_low_alpha, f"Error did not decrease with higher alpha: {err_low_alpha:.4f} vs {err_high_alpha:.4f}"

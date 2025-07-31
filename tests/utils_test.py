import pytest
import matplotlib
matplotlib.use("Agg")
from dame_bs.utils import theoretical_upper_bound,plot_errorbars
import numpy as np
from dame_bs.dame_bs import dame_with_binary_search
import matplotlib.pyplot as plt
from experiments.univariate_experiment import compare_univariate_algorithms

########################################################################################

def test_theoretical_upper_bound_valid():
    '''Verifies that theoretical_upper_bound returns a non-negative float.'''
    alpha = 0.6
    n = 9000
    m = 20
    bound = theoretical_upper_bound(alpha, n, m)
    assert bound >= 0
    assert isinstance(bound, float)



###########################################################################################
def test_estimate_within_error_and_bound():
    '''Ensures that the DAME-BS empirical error is less than theoretical bound.'''

    np.random.seed(42)

    true_mean = 0.3
    n = 8000
    m = 20
    alpha = 0.6

    median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,"normal",true_mean,trials=50)
    # Theoretical bound check
    theoretical_bound = theoretical_upper_bound(alpha, n, m)  
    assert median_dame <= theoretical_bound, f"Error {median_dame:.4f} exceeds theoretical bound {theoretical_bound:.4f}"

#############################################################################################

def test_theoretical_bound_no_noise():
    '''Tests the edge case where alpha = ∞ (i.e., no added noise), 
    and ensures the theoretical bound is reduced to its minimal form.'''

    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    pi_alpha = 1
    n=1000
    m = 20
    true_mean = 0.3
    
    median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,"normal",true_mean,trials=50)
    theoretical_bound=theoretical_upper_bound(alpha, n, m)
    print(theoretical_bound)
    assert median_dame < 1e-4, f"Error is unexpectedly high without privacy: {median_dame:.6f}"
    assert theoretical_bound == 32*n*np.exp(-n/2),f"Theoretical Upper bound still contains error due to noise"

##################################################################################################

def test_bound_decreases_with_n():
    """As n increases (with alpha,m fixed), the bound should decrease."""
    alpha=0.6
    n=1000
    m=20
    b_small = theoretical_upper_bound(alpha, n, m)
    b_large = theoretical_upper_bound(alpha, n*2, m)
    assert b_large <= b_small

##################################################################################################

def test_bound_decreases_with_m():
    """As m increases (with alpha,n fixed), the bound should decrease."""
    n = 500
    alpha = 0.6
    b_small = theoretical_upper_bound(alpha, n, 5)
    b_large = theoretical_upper_bound(alpha, n, 200)
    assert b_large <= b_small



##################################################################################################
def test_invalid_parameters():
    """Negative or zero alpha should still produce a float or raise as appropriate."""
    with pytest.raises(ValueError):
        theoretical_upper_bound(-0.6, 1000, 4)
    with pytest.raises(ValueError):
        theoretical_upper_bound(0.6, -1000, 4)
    with pytest.raises(ValueError):
        theoretical_upper_bound(0.6, 1000, -4)


##################################################################################################

def test_plot_errorbars_runs(monkeypatch):
    '''Ensures the plotting function executes without crashing. 
    Uses monkeypatch to suppress actual plot display.'''

    # Monkeypatch plt.show to avoid displaying the plot during test
    monkeypatch.setattr(plt, "show", lambda: None)

    true_mean = 0.3
    n = 9000
    m = 20
    median_errors_dame = []
    lower_errors_dame = []
    upper_errors_dame = []
    median_errors_kent = []
    lower_errors_kent = []
    upper_errors_kent = []
    alphas=np.linspace(0.1, 1.0, 3)
    distribution="normal"

    for alpha in alphas:
        
        
        median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=10)
            
        median_errors_dame.append(median_dame)
        lower_errors_dame.append(lower_dame)
        upper_errors_dame.append(upper_dame)
        median_errors_kent.append(median_kent)
        lower_errors_kent.append(lower_kent)
        upper_errors_kent.append(upper_kent)

    
    plot_errorbars(alphas, median_errors_kent,median_errors_dame, lower_errors_kent,
                   lower_errors_dame,upper_errors_kent, upper_errors_dame, "Privacy parameter α", 
                   ylabel="Mean Squared Error", title=f"Mean Squared Error vs Alpha for the {distribution} distribution",
                   log_scale=True,plot_ub=False,upper_bounds=None)
    



##################################################################################################

def test_plot_errorbars_runs_with_ub(monkeypatch):
    '''Ensures the plotting function executes with upper bounds as well. 
    Uses monkeypatch to suppress actual plot display.'''

    # Monkeypatch plt.show to avoid displaying the plot during test
    monkeypatch.setattr(plt, "show", lambda: None)

    true_mean = 0.3
    n = 9000
    m = 20
    median_errors_dame = []
    lower_errors_dame = []
    upper_errors_dame = []
    median_errors_kent = []
    lower_errors_kent = []
    upper_errors_kent = []
    alphas=np.linspace(0.1, 1.0, 3)
    distribution="normal"

    for alpha in alphas:
        
        
        median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=10)
            
        median_errors_dame.append(median_dame)
        lower_errors_dame.append(lower_dame)
        upper_errors_dame.append(upper_dame)
        median_errors_kent.append(median_kent)
        lower_errors_kent.append(lower_kent)
        upper_errors_kent.append(upper_kent)

    
    
    upper_bounds = [theoretical_upper_bound(alpha, n, m) for alpha in alphas]

    plot_errorbars(alphas, median_errors_kent,median_errors_dame, lower_errors_kent,
                   lower_errors_dame,upper_errors_kent, upper_errors_dame, "Privacy parameter α", 
                   ylabel="Mean Squared Error", title=f"Mean Squared Error vs Alpha for the {distribution} distribution",
                   log_scale=True,plot_ub=False,upper_bounds=upper_bounds)
    



import pytest
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

    mean_dame,_,_,_= compare_univariate_algorithms(n,m,alpha,"normal",true_mean,trials=50)
    # Theoretical bound check
    theoretical_bound = theoretical_upper_bound(alpha, n, m)  
    assert mean_dame <= theoretical_bound, f"Error {mean_dame:.4f} exceeds theoretical bound {theoretical_bound:.4f}"

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
    
    mean_dame,_,_,_= compare_univariate_algorithms(n,m,alpha,"normal",true_mean,trials=50)
    theoretical_bound=theoretical_upper_bound(alpha, n, m)
    print(theoretical_bound)
    assert mean_dame < 1e-4, f"Error is unexpectedly high without privacy: {mean_dame:.6f}"
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
    mean_errors_kent = []
    std_errors_kent = []
    mean_errors_dame = []
    std_errors_dame = []
    alphas=np.linspace(0.1, 1.0, 3)
    distribution="normal"

    for alpha in alphas:
        
        
        mean_dame,std_dame,mean_kent,std_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=50)
            
        mean_errors_kent.append(mean_kent)
        std_errors_kent.append(std_kent)
        mean_errors_dame.append(mean_dame)
        std_errors_dame.append(std_dame)

    plot_errorbars(alphas, mean_errors_kent,mean_errors_dame, std_errors_kent,
                   std_errors_dame, "Privacy parameter α", ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Alpha for the {distribution} distribution",
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
    mean_errors_kent = []
    std_errors_kent = []
    mean_errors_dame = []
    std_errors_dame = []
    alphas=np.linspace(0.1, 1.0, 3)
    distribution="normal"

    for alpha in alphas:
        
        
        mean_dame,std_dame,mean_kent,std_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=50)
            
        mean_errors_kent.append(mean_kent)
        std_errors_kent.append(std_kent)
        mean_errors_dame.append(mean_dame)
        std_errors_dame.append(std_dame)
    
    upper_bounds = [theoretical_upper_bound(alpha, n, m) for alpha in alphas]

    plot_errorbars(alphas, mean_errors_kent,mean_errors_dame, std_errors_kent,
                   std_errors_dame, "Privacy parameter α", ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Alpha for the {distribution} distribution",
                   log_scale=True,plot_ub=True,upper_bounds=upper_bounds)



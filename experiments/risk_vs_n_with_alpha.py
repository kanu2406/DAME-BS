"""
experiments.risk_vs_n_with_alpha
---------------------

This script runs the DAME-BS algorithm for various distributions and plots
the mean squared error (MSE) vs. number of users `n` for different values of alphas.

It uses utility functions from `dame_bs.utils`.


Usage:
    Run directly to generate plots.

"""



import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.utils import run_dame_experiment, plot_errorbars_and_upper_bounds,theoretical_upper_bound

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.utils import run_dame_experiment, theoretical_upper_bound

def experiment_risk_vs_n_multiple_alphas(distribution="normal", 
                                         alpha_values=[np.inf, 1.0, 0.5,0.2], 
                                         min_n=500, 
                                         max_n=15000, 
                                         step=500, 
                                         m=20, 
                                         true_mean=0.3, 
                                         trials=50, 
                                         delta=0.1):
    
    """
    Args:
        distribution (str):Name of the distribution ( "normal", "uniform", "poisson","exponential").
        alpha_values (List[float]):List of privacy/risk parameters α to compare. Use `np.inf` to indicate the non-private (infinite-α) case.
        min_n (int):Smallest user population size to test.Default is 500.
        max_n (int):Largest user population size to test. Default id 15000.
        step (int):Increment in n between successive experiment points. Default is 500.
        m (int): Number of samples per user. Default is 20.
        true_mean (float): True mean value used for the experiment. Default is 0.3.
        trials (int): Number of experiment trials. Default is 50.
        delta (float): tolerated probability of failure of binary search in DAME-BS. Default is 0.1.

    Returns:
        None. Displays plots.
    """
    
    n_values = list(range(min_n, max_n + 1, step))
    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        mean_errors = []
        std_errors = []

        for n in n_values:
            mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials, distribution, delta)
            mean_errors.append(mean_err)
            std_errors.append(std_err)

        # Plot empirical errors
        label = r"$\alpha = \infty$" if alpha == np.inf else fr"$\alpha = {alpha}$"
        plt.errorbar(n_values, mean_errors, yerr=std_errors, label=f"MSE ({label})", marker='o', capsize=3)

        # Plot theoretical bound
        upper_bounds = [theoretical_upper_bound(alpha, n, m) for n in n_values]
        plt.plot(n_values, upper_bounds, '--', label=f"Theory UB ({label})")

    # Final plot settings
    plt.xlabel("Number of users (n)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"MSE vs n for {distribution} distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    distributions = ["normal", "uniform", "poisson", "exponential"]
    # distributions = ["poisson"]
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_n_multiple_alphas(distribution=dist)


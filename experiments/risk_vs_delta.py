"""
experiments.risk_vs_delta
--------------------------

This script runs the DAME-BS algorithm for various distributions and plots
the mean squared error (MSE) vs. failure probability of binary search delta.

It imports core utilities from `dame_bs.utils` and supports multiple distributions.

Usage:
    Run directly to generate plots.


"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.utils import run_dame_experiment, plot_errorbars_and_upper_bounds,theoretical_upper_bound


def experiment_risk_vs_delta_for_dist(distribution,n=9000, m=20, true_mean=0.3, trials=50,alpha=0.6):
    """
    Runs a risk-vs-delta experiment for a given distribution.

    Args:
        distribution (str): Name of the distribution ( "normal", "uniform","poisson","exponential").
        n (int): Number of users. Default is 9000.
        m (int): Number of samples per user. Default is 20.
        true_mean (float): True mean value used for the experiment. Default is 0.3.
        trials (int): Number of experiment trials. Default is 50.
        alpha (float): Privacy parameter. Default is 0.6.

    Returns:
        None. Displays plots for each distribution.
    """
    
    delta_values = np.linspace(0.1, 1.0, 20)
    mean_errors = []
    std_errors = []
    deltas=[]
    

    for delta in delta_values:
        
        deltas.append(delta)
        mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials,distribution,delta)
        mean_errors.append(mean_err)
        std_errors.append(std_err)

    # upper_bounds = [theoretical_upper_bound(alpha, n, m,delta) for delta in deltas]

    # plot_errorbars_and_upper_bounds(deltas, mean_errors, std_errors,upper_bounds, 
    #                xlabel="Tolerated Failure Probability delta"
    #                , ylabel="Mean Squared Error", 
    #                title=f"Mean Squared Error vs Delta for the {distribution} distribution")
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(deltas, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.title(f"Mean Squared Error vs Delta for the {distribution} distribution")
    plt.xlabel("Tolerated Failure Probability delta")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    distributions = ["normal", "uniform", "poisson", "exponential"]
    # distributions = ["poisson"]
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_delta_for_dist(dist)
        
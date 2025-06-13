import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_ts.utils import run_dame_experiment, plot_errorbars_and_upper_bounds,theoretical_upper_bound


def experiment_risk_vs_n_for_dist(distribution,alpha=0.6,min_n=5000, m=20, true_mean=0.3, trials=50):
    n_values = list(range(min_n, 20000 + 1, 2000))
    mean_errors = []
    std_errors = []
    

    for n in n_values:
        
        mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials,distribution)
        mean_errors.append(mean_err)
        std_errors.append(std_err)

    upper_bounds = [theoretical_upper_bound(alpha, n, m) for n in n_values]

    plot_errorbars_and_upper_bounds(n_values, mean_errors, std_errors,upper_bounds, 
                   xlabel="Number of users (n)"
                   , ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Number of Users (n) for the {distribution} distribution")

if __name__ == "__main__":
    distributions = ["normal", "uniform", "laplace", "exponential"]
    for dist in distributions:
        experiment_risk_vs_n_for_dist(dist)

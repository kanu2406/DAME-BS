import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_ts.utils import run_dame_experiment, plot_errorbars_and_upper_bounds,theoretical_upper_bound


def experiment_risk_vs_alpha_for_dist(distribution,n=5000, m=20, true_mean=0.3, trials=50):
    alpha_values = np.linspace(0.6, 2.0, 20)
    mean_errors = []
    std_errors = []
    alphas=[]
    

    for alpha in alpha_values:
        
        alphas.append(alpha)
        mean_err, std_err = run_dame_experiment(n, alpha, m, true_mean, trials,distribution)
        mean_errors.append(mean_err)
        std_errors.append(std_err)

    upper_bounds = [theoretical_upper_bound(a, n, m) for a in alphas]

    plot_errorbars_and_upper_bounds(alphas, mean_errors, std_errors,upper_bounds, 
                   xlabel="Privacy parameter α"
                   , ylabel="Mean Squared Error", 
                   title=f"Mean Squared Error vs Alpha for the {distribution} distribution")

if __name__ == "__main__":
    distributions = ["normal", "uniform", "laplace", "exponential"]
    for dist in distributions:
        experiment_risk_vs_alpha_for_dist(dist)

    # experiment_risk_vs_alpha_for_dist("normal")

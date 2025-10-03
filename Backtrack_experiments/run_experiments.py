import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Backtrack_experiments.univariate_experiments import *


def main():
    
    distributions = [ "normal","uniform","binomial","standard_t"]
    # distributions = ["normal","uniform"]
    n_values = list(range(40, 20000  , 1000))
    m_values = list(range(80, 6000, 500))
    alpha_values = np.linspace(0.05, 4.0,15 )
    true_mean = 0.1
    fixed_n = 10000
    fixed_m = 1000
    fixed_alpha = 0.6
    base_seed = 42
    
    print("Running univariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"Backtrack_experiments/results/results_univariate/mse_vs_alpha_{dist}.csv"
        res = run_param("alpha",alpha_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=20,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
        res["param_values"],res["median_dame"],
        res["lower10_dame"],res["upper90_dame"],
        res["median_dame_backtrack"],res["lower10_dame_backtrack"],res["upper90_dame_backtrack"],
        xlabel="alpha",
        ylabel="Median Squared Error",
        title=f"Median Squared Error vs alpha for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"Backtrack_experiments/results/plots_univariate/mse_vs_alpha_{dist}.png",
        log_log_scale=True,
        y_lim=True
    )

    
    print("Running univariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"Backtrack_experiments/results/results_univariate/mse_vs_n_{dist}.csv"
        res = run_param("n",n_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
         res["param_values"],res["median_dame"],
        res["lower10_dame"],res["upper90_dame"],
        res["median_dame_backtrack"],res["lower10_dame_backtrack"],res["upper90_dame_backtrack"],
        xlabel="n",
        ylabel="Mean Squared Error",
        title=f"MSE vs n for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"Backtrack_experiments/results/plots_univariate/mse_vs_n_{dist}.png",
        log_log_scale=True
    )
        


    print("Running univariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    
    for dist in distributions:
        
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"Backtrack_experiments/results/results_univariate/mse_vs_m_{dist}.csv"
        res = run_param("m",m_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=20,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
         res["param_values"],res["median_dame"],
        res["lower10_dame"],res["upper90_dame"],
        res["median_dame_backtrack"],res["lower10_dame_backtrack"],res["upper90_dame_backtrack"],
        xlabel="m",
        ylabel="Mean Squared Error",
        title=f"MSE vs m for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"Backtrack_experiments/results/plots_univariate/mse_vs_m_{dist}.png",
        log_log_scale=True
    )  
    
if __name__ == "__main__":
    main()

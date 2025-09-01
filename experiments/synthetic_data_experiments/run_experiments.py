import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from experiments.synthetic_data_experiments.univariate_experiment import *
from experiments.synthetic_data_experiments.multivariate_experiment import *


def main():
    """
    Main execution function that runs a series of experiments across:
    - Univariate distributions (Normal, Uniform, Binomial, Standard_t)
    - Multivariate distributions with varying dimensions

    It compares three private mean estimation algorithms (DAME-BS, Kent, Girgis)

    Each distribution is evaluated under variation for different value of following parameters:
    - `alpha` (privacy parameter)
    - `n` (number of users)
    - `m` (number of samples per user)
    - `d` (dimensionality)

    The true mean used for comparison is fixed (scalar for univariate, vector for multivariate).
    Each experiment is repeated for a fixed number of trials = 50.
    """
    distributions = [ "normal","uniform","binomial","standard_t"]
    # distributions = ["binomial"]
    n_values = list(range(40, 20000  , 1000))
    n_values_multi = list(range(40, 20000  , 2000))
    m_values = list(range(80, 6000, 400))
    m_values_multi = list(range(5, 1000, 70))
    alpha_values = np.linspace(0.05, 4.0,4 )
    alpha_values_multi = np.linspace(0.1, 6.0,20 )
    d_values = [2,4,8,16,32,64,256]
    true_mean = 0.7
    fixed_n = 2000
    fixed_m = 20
    fixed_alpha = 0.6
    fixed_d = 8
    fixed_true_mean = [0.7]*8
    base_seed = 42
    
    print("Running univariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        if dist=="binomial":
            log_log_scale= True
        else:
            log_log_scale=False
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_univariate/mse_vs_alpha_{dist}.csv"
        res = run_param("alpha",alpha_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=40,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
        res["param_values"],
        res["median_kent"], res["median_dame"],
        res["lower10_kent"], res["lower10_dame"],
        res["upper90_kent"], res["upper90_dame"],
        res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
        xlabel="alpha",
        ylabel="Median Squared Error",
        title=f"Median Squared Error vs alpha for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_alpha_{dist}.png",
        log_log_scale=log_log_scale,
        y_lim=True
    )

    
    print("Running univariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_univariate/mse_vs_n_{dist}.csv"
        res = run_param("n",n_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
        res["param_values"],
        res["median_kent"], res["median_dame"],
        res["lower10_kent"], res["lower10_dame"],
        res["upper90_kent"], res["upper90_dame"],
        res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
        xlabel="n",
        ylabel="Mean Squared Error",
        title=f"MSE vs n for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_n_{dist}.png",
        log_log_scale=True
    )
    print("Running univariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    
    for dist in distributions:
        
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_univariate/mse_vs_m_{dist}.csv"
        res = run_param("m",m_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
        base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
        plot_errorbars(
        res["param_values"],
        res["median_kent"], res["median_dame"],
        res["lower10_kent"], res["lower10_dame"],
        res["upper90_kent"], res["upper90_dame"],
        res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
        xlabel="m",
        ylabel="Mean Squared Error",
        title=f"MSE vs m for {dist} distribution",
        log_scale=True,
        plot_ub=False,
        upper_bounds=None,
        save_path=f"experiments/synthetic_data_experiments/results/plots_univariate/mse_vs_m_{dist}.png",
        log_log_scale=True
    )  
    
    print("Running multivariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_multivariate/mse_vs_alpha_{dist}.csv"
        res = run_param_multivariate(
        param_name="alpha",
        param_values=alpha_values_multi,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=30,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
            xlabel="alpha",
            ylabel="Median Squared Error",
            title=f"Median Squared l_2 error vs alpha for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_alpha_{dist}.png",
            log_log_scale=False
        )
        

    print("Running multivariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_multivariate/mse_vs_n_{dist}.csv"
        res = run_param_multivariate(
        param_name="n",
        param_values=n_values_multi,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=30,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
            xlabel="n",
            ylabel="Median Squared l_2 Error",
            title=f"Median squared l_2 error vs n for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_n_{dist}.png"
        )                           
    
    print("Running multivariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_multivariate/mse_vs_m_{dist}.csv"
        res = run_param_multivariate(
        param_name="m",
        param_values=m_values_multi,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=30,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
            xlabel="m",
            ylabel="Median squared l_2 error",
            title=f"Median squared l_2 error vs m for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_m_{dist}.png",
            log_log_scale=True
        )  

    print("Running univariate experiments for different distributions for various values of d.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results/results_multivariate/mse_vs_d_{dist}.csv"
        res = run_param_multivariate(
        param_name="d",
        param_values=d_values,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=30,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            res["median_girgis"],res["lower10_girgis"],res["upper90_girgis"],
            xlabel="d",
            ylabel="Mean Squared Error",
            title=f"MSE vs d for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/results/plots_multivariate/mse_vs_d_{dist}.png"
        )
    

if __name__ == "__main__":
    main()

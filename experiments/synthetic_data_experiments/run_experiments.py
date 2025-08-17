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
    - Univariate distributions (Normal, Uniform, Binomial)
    - Multivariate distributions with varying dimensions

    Each distribution is evaluated under variation for different value of following parameters:
    - `alpha` (privacy parameter)
    - `n` (number of users)
    - `m` (number of samples per user)
    - `d` (dimensionality)

    The true mean used for comparison is fixed (scalar for univariate, vector for multivariate).
    Each experiment is repeated for a fixed number of trials = 50.
    """
    distributions = ["normal", "uniform","binomial","standard_t"]
    # distributions = ["uniform"]
    n_values = list(range(40, 10000  , 400))
    m_values = list(range(1, 500, 40))
    alpha_values = np.linspace(0.1, 3.0,20 )
    d_values = list(range(2,200,30))
    true_mean = 0.3
    fixed_n = 8000
    fixed_m = 40
    fixed_alpha = 0.6
    fixed_d = 8
    fixed_true_mean = [0.3]*8
    base_seed = 42
    
    # print("Running univariate experiments for different distributions for various alpha values.")
    # print("------------------------------------------------------------------------------------")
    # for dist in distributions:
    #     print(f"\n Running experiment for distribution: {dist}")
    #     out_path=f"experiments/synthetic_data_experiments/results_univariate/mse_vs_alpha_{dist}.csv"
    #     res = run_param("alpha",alpha_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
    #     base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
    #     plot_errorbars(
    #     res["param_values"],
    #     res["median_kent"], res["median_dame"],
    #     res["lower10_kent"], res["lower10_dame"],
    #     res["upper90_kent"], res["upper90_dame"],
    #     xlabel="alpha",
    #     ylabel="Mean Squared Error",
    #     title=f"MSE vs alpha for {dist} distribution",
    #     log_scale=True,
    #     plot_ub=False,
    #     upper_bounds=None,
    #     save_path=f"experiments/synthetic_data_experiments/plots_univariate/mse_vs_alpha_{dist}.png"
    # )

    
    # print("Running univariate experiments for different distributions for various values of n.")
    # print("------------------------------------------------------------------------------------")
    # for dist in distributions:
    #     print(f"\n Running experiment for distribution: {dist}")
    #     out_path=f"experiments/synthetic_data_experiments/results_univariate/mse_vs_n_{dist}.csv"
    #     res = run_param("n",n_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
    #     base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
    #     plot_errorbars(
    #     res["param_values"],
    #     res["median_kent"], res["median_dame"],
    #     res["lower10_kent"], res["lower10_dame"],
    #     res["upper90_kent"], res["upper90_dame"],
    #     xlabel="n",
    #     ylabel="Mean Squared Error",
    #     title=f"MSE vs n for {dist} distribution",
    #     log_scale=True,
    #     plot_ub=False,
    #     upper_bounds=None,
    #     save_path=f"experiments/synthetic_data_experiments/plots_univariate/mse_vs_n_{dist}.png"
    # )
    # print("Running univariate experiments for different distributions for various values of m.")
    # print("------------------------------------------------------------------------------------")
    # for dist in distributions:
    #     print(f"\n Running experiment for distribution: {dist}")
    #     out_path=f"experiments/synthetic_data_experiments/results_univariate/mse_vs_m_{dist}.csv"
    #     res = run_param("m",m_values,fixed_n,fixed_m,fixed_alpha,dist,true_mean,trials_per_setting=50,
    #     base_seed=base_seed,out_csv_path=out_path,n_jobs=8)
    #     plot_errorbars(
    #     res["param_values"],
    #     res["median_kent"], res["median_dame"],
    #     res["lower10_kent"], res["lower10_dame"],
    #     res["upper90_kent"], res["upper90_dame"],
    #     xlabel="m",
    #     ylabel="Mean Squared Error",
    #     title=f"MSE vs m for {dist} distribution",
    #     log_scale=True,
    #     plot_ub=False,
    #     upper_bounds=None,
    #     save_path=f"experiments/synthetic_data_experiments/plots_univariate/mse_vs_m_{dist}.png"
    # )  
    
    # print("Running multivariate experiments for different distributions for various alpha values.")
    # print("------------------------------------------------------------------------------------")
    # for dist in distributions:
    #     print(f"\n Running experiment for distribution: {dist}")
    #     out_path=f"experiments/synthetic_data_experiments/results_multivariate/mse_vs_alpha_{dist}.csv"
    #     res = run_param_multivariate(
    #     param_name="alpha",
    #     param_values=alpha_values,
    #     fixed_n=fixed_n,
    #     fixed_m=fixed_m,
    #     fixed_alpha=fixed_alpha,     
    #     fixed_d=fixed_d,
    #     distribution=dist,
    #     true_mean=fixed_true_mean,
    #     trials_per_setting=50,
    #     base_seed=base_seed,
    #     out_csv_path=out_path,
    #     n_jobs=8)

    #     plot_errorbars(
    #         res["param_values"],
    #         res["median_kent"], res["median_dame"],
    #         res["lower10_kent"], res["lower10_dame"],
    #         res["upper90_kent"], res["upper90_dame"],
    #         xlabel="alpha",
    #         ylabel="Mean Squared Error",
    #         title=f"MSE vs alpha for {dist} distribution",
    #         log_scale=True,
    #         plot_ub=False,
    #         upper_bounds=None,
    #         save_path=f"experiments/synthetic_data_experiments/plots_multivariate/mse_vs_alpha_{dist}.png"
    #     )
        

    print("Running multivariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results_multivariate/mse_vs_n_{dist}.csv"
        res = run_param_multivariate(
        param_name="n",
        param_values=n_values,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=50,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            xlabel="n",
            ylabel="Mean Squared Error",
            title=f"MSE vs n for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/plots_multivariate/mse_vs_n_{dist}.png"
        )                           
    
    print("Running multivariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results_multivariate/mse_vs_m_{dist}.csv"
        res = run_param_multivariate(
        param_name="m",
        param_values=m_values,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=50,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            xlabel="m",
            ylabel="Mean Squared Error",
            title=f"MSE vs m for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/plots_multivariate/mse_vs_m_{dist}.png"
        )  

    print("Running univariate experiments for different distributions for various values of d.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        out_path=f"experiments/synthetic_data_experiments/results_multivariate/mse_vs_d_{dist}.csv"
        res = run_param_multivariate(
        param_name="d",
        param_values=d_values,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        fixed_alpha=fixed_alpha,     
        fixed_d=fixed_d,
        distribution=dist,
        true_mean=fixed_true_mean,
        trials_per_setting=50,
        base_seed=base_seed,
        out_csv_path=out_path,
        n_jobs=8)

        plot_errorbars(
            res["param_values"],
            res["median_kent"], res["median_dame"],
            res["lower10_kent"], res["lower10_dame"],
            res["upper90_kent"], res["upper90_dame"],
            xlabel="d",
            ylabel="Mean Squared Error",
            title=f"MSE vs d for {dist} distribution",
            log_scale=True,
            plot_ub=False,
            upper_bounds=None,
            save_path=f"experiments/synthetic_data_experiments/plots_multivariate/mse_vs_d_{dist}.png"
        )
    

if __name__ == "__main__":
    main()

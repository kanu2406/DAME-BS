import pandas as pd
import numpy as np
import time,random
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from experiments.real_data_experiments.mimic.preprocess import *
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator
from collections import defaultdict
from girgis.scalar import *

np.random.seed(42)
def main():
    """
    This script runs a 500-trial comparison of Kent's mean estimator, algorithm proposed by Girgis and DAME-BS
    on real heart rate data from the MIMIC-III dataset.

    Steps:
        - Loads and filters MIMIC-III CHARTEVENTS for heart rate data.
        - Scales all values to [-1, 1] and truncates samples so each user have same number of samples.
        - Runs all three algorithms across 500 trials.
        - Reports runtime, mean estimates, and median MSE (in both scaled and original ranges) including 10th and 90th percentile.
        - Saves each trial along with other parameter in a csv file.
        - Also saves the final summary of results.

    """

    # Load patients data 
    df = pd.read_csv("experiments/Datasets/CHARTEVENTS.csv", low_memory=False)

    # Filter for heart rate (i.e. itemid = 211) and non-missing values
    df_hr = df[(df['itemid'] == 211) & (df['valuenum'].notna())]
    # Keep only relevant columns
    df_hr = df_hr[['subject_id','itemid', 'charttime', 'valuenum']]
    print(df_hr.head())

    # converting to dictionary
    user_samples = defaultdict(list)
    for row in df_hr.itertuples():
        sid = row.subject_id
        if sid not in user_samples:
            user_samples[sid] = []
        user_samples[sid].append(row.valuenum)
    
    # keeping only SIDs with at least 7 samples
    user_samples = {sid: vals for sid, vals in user_samples.items() if len(vals) >= 7}

    print("Data Loaded.")

    # Scaling everything to range [-1,1]
    user_samples_scaled,desired_length,true_mean,true_mean_scaled,vmin,vmax=scaling_data(user_samples)
    print("Scaling Done.")

    # Truncating and shuffling 
    users_and_heart_rate_final = truncate_and_shuffle(user_samples_scaled,desired_length)
    print("Truncation Done.")
    print("Total number of users", len(user_samples))
    print("Number of samples per user : ",desired_length)
    print(f"True mean : {true_mean:.3f}" )
    print(f"True mean scaled : {true_mean_scaled:.3f}" )
    
    
    num_exp=500
    print("Running both algorithms 500 times")
    # Flattening the data to list 
    heart_rates=[vals for vals in users_and_heart_rate_final.values()]
    n = len(heart_rates)
    m = desired_length
    alpha = 0.6

    base_seed = 42             
    out_trials_csv = "experiments/real_data_experiments/mimic/results/mimic_trials.csv"
    out_summary_csv = "experiments/real_data_experiments/mimic/results/mimic_summary.csv"
    
    # initialize output CSV
    init_results_csv(out_trials_csv)

    print(f"Running both algorithms for {num_exp} trials; writing to {out_trials_csv}")

    for trial in range(num_exp):
        seed = make_seed(base_seed, trial)
        np.random.seed(seed)
        random.seed(seed)
        rates=heart_rates.copy()
        np.random.shuffle(rates)
        X=np.array(rates)

        try:
            # Kent
            t0 = time.time()
            theta_hat_kent_scaled = kent_mean_estimator(X, alpha=alpha, K=1.0)
            t1 = time.time()
            time_k = t1 - t0
            theta_hat_kent_orig = 0.5 * (theta_hat_kent_scaled + 1) * (vmax - vmin) + vmin

            # DAME-BS
            t0 = time.time()
            theta_hat_dame_scaled = dame_with_binary_search(n, alpha, m, X)
            t1 = time.time()
            time_d = t1 - t0
            theta_hat_dame_orig = 0.5 * (theta_hat_dame_scaled + 1) * (vmax - vmin) + vmin

            # Girgis
            gamma = 0.05
            tau = np.sqrt((np.log(2*n/gamma))/m)
            inv_tau = 1/tau

            # Round to nearest power of 2
            nearest_pow2 = 2**int(np.round(np.log2(inv_tau)))

            # Adjusted tau
            tau_adj = 1/nearest_pow2
            t0 = time.time()
            theta_hat_girgis_scaled = meanscalar(X, tau_adj, alpha, B=1.0)
            t1 = time.time()
            time_g = t1 - t0
            theta_hat_girgis_orig = 0.5 * (theta_hat_girgis_scaled + 1) * (vmax - vmin) + vmin


            # errors (scaled and original)
            scaled_mse_kent = (theta_hat_kent_scaled - true_mean_scaled) ** 2
            scaled_mse_dame = (theta_hat_dame_scaled - true_mean_scaled) ** 2
            scaled_mse_girgis = (theta_hat_girgis_scaled - true_mean_scaled) ** 2
            orig_mse_kent = (theta_hat_kent_orig - true_mean) ** 2
            orig_mse_dame = (theta_hat_dame_orig - true_mean) ** 2
            orig_mse_girgis = (theta_hat_girgis_orig - true_mean) ** 2

            status = "ok"

        except Exception as e:
            # on error, record NaNs and error message
            theta_hat_kent_scaled = np.nan
            theta_hat_dame_scaled = np.nan
            theta_hat_girgis_scaled = np.nan
            theta_hat_kent_orig = np.nan
            theta_hat_dame_orig = np.nan
            theta_hat_girgis_orig = np.nan
            scaled_mse_kent = np.nan
            scaled_mse_dame = np.nan
            scaled_mse_girgis = np.nan
            orig_mse_kent = np.nan
            orig_mse_dame = np.nan
            orig_mse_girgis = np.nan
            time_k = np.nan
            time_d = np.nan
            time_g = np.nan
            status = f"error: {repr(e)}"
            print(f"Trial {trial}: error: {repr(e)}")

        # prepare row and append 
        row = {
            "trial": int(trial),
            "seed": int(seed),
            "n": int(n),
            "m": int(m),
            "alpha": float(alpha),
            "theta_hat_kent_scaled": float(theta_hat_kent_scaled) if not np.isnan(theta_hat_kent_scaled) else np.nan,
            "theta_hat_dame_scaled": float(theta_hat_dame_scaled) if not np.isnan(theta_hat_dame_scaled) else np.nan,
            "theta_hat_girgis_scaled": float(theta_hat_girgis_scaled) if not np.isnan(theta_hat_girgis_scaled) else np.nan,
            "theta_hat_kent_orig": float(theta_hat_kent_orig) if not np.isnan(theta_hat_kent_orig) else np.nan,
            "theta_hat_dame_orig": float(theta_hat_dame_orig) if not np.isnan(theta_hat_dame_orig) else np.nan,
            "theta_hat_girgis_orig": float(theta_hat_girgis_orig) if not np.isnan(theta_hat_girgis_orig) else np.nan,
            "scaled_mse_kent": float(scaled_mse_kent) if not np.isnan(scaled_mse_kent) else np.nan,
            "scaled_mse_dame": float(scaled_mse_dame) if not np.isnan(scaled_mse_dame) else np.nan,
            "scaled_mse_girgis": float(scaled_mse_girgis) if not np.isnan(scaled_mse_girgis) else np.nan,
            "orig_mse_kent": float(orig_mse_kent) if not np.isnan(orig_mse_kent) else np.nan,
            "orig_mse_dame": float(orig_mse_dame) if not np.isnan(orig_mse_dame) else np.nan,
            "orig_mse_girgis": float(orig_mse_girgis) if not np.isnan(orig_mse_girgis) else np.nan,
            "time_kent_s": float(time_k) if not np.isnan(time_k) else np.nan,
            "time_dame_s": float(time_d) if not np.isnan(time_d) else np.nan,
            "time_girgis_s": float(time_g) if not np.isnan(time_g) else np.nan,
            "status": status
        }
        append_row_csv(out_trials_csv, row)

    print(f"Trials complete. Aggregating results to {out_summary_csv}")

    # Aggregate and save summary 
    df = pd.read_csv(out_trials_csv)

    # compute summary
    summary = {
        "dataset": "MIMIC_Heart_Rate_Data",
        "n": n,
        "m": m,
        "alpha": alpha,
        "trials": num_exp,
        "median_scaled_mse_kent": float(df["scaled_mse_kent"].median(skipna=True)),
        "median_scaled_mse_dame": float(df["scaled_mse_dame"].median(skipna=True)),
        "median_scaled_mse_girgis": float(df["scaled_mse_girgis"].median(skipna=True)),
        "10pct_scaled_mse_kent": float(df["scaled_mse_kent"].quantile(0.1)),
        "90pct_scaled_mse_kent": float(df["scaled_mse_kent"].quantile(0.9)),
        "10pct_scaled_mse_dame": float(df["scaled_mse_dame"].quantile(0.1)),
        "90pct_scaled_mse_dame": float(df["scaled_mse_dame"].quantile(0.9)),
        "10pct_scaled_mse_girgis": float(df["scaled_mse_girgis"].quantile(0.1)),
        "90pct_scaled_mse_girgis": float(df["scaled_mse_girgis"].quantile(0.9)),
        "median_orig_mse_kent": float(df["orig_mse_kent"].median(skipna=True)),
        "median_orig_mse_dame": float(df["orig_mse_dame"].median(skipna=True)),
        "median_orig_mse_girgis": float(df["orig_mse_girgis"].median(skipna=True)),
        "mean_time_kent_s": float(df["time_kent_s"].mean(skipna=True)),
        "mean_time_dame_s": float(df["time_dame_s"].mean(skipna=True)),
        "mean_time_girgis_s": float(df["time_girgis_s"].mean(skipna=True)),
        }
    
    # save summary
    make_file(out_summary_csv)
    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)

    print("Summary saved. Done.\n")
    print("-" * 60)
    print(f"Dataset:               {summary['dataset']}")
    print(f" Privacy alpha:        {summary['alpha']}")
    print(f" Trials:               {summary['trials']}")
    print("-" * 60)
    print(" Mean Squared Error (Scaled values)")
    print(f"   Kent median MSE:   {summary['median_scaled_mse_kent']:.4e} "
          f"(10%: {summary['10pct_scaled_mse_kent']:.2e}, 90%: {summary['90pct_scaled_mse_kent']:.2e})")
    print(f"   DAME median MSE:   {summary['median_scaled_mse_dame']:.4e} "
          f"(10%: {summary['10pct_scaled_mse_dame']:.2e}, 90%: {summary['90pct_scaled_mse_dame']:.2e})")
    print(f"   Girgis median MSE:   {summary['median_scaled_mse_girgis']:.4e} "
          f"(10%: {summary['10pct_scaled_mse_girgis']:.2e}, 90%: {summary['90pct_scaled_mse_girgis']:.2e})")
    print("-" * 60)
    print(" Mean Squared Error (Original values)")
    print(f"   Kent median MSE:   {summary['median_orig_mse_kent']:.4e}")
    print(f"   DAME median MSE:   {summary['median_orig_mse_dame']:.4e}")
    print(f"   Girgis median MSE:   {summary['median_orig_mse_girgis']:.4e}")
    print("-" * 60)
    
    print(" Runtime (seconds)")
    print(f"   Kent mean runtime: {summary['mean_time_kent_s']:.4f}")
    print(f"   DAME mean runtime: {summary['mean_time_dame_s']:.4f}")
    print(f"   Girgis mean runtime: {summary['mean_time_girgis_s']:.4f}")
    print("-" * 60)

if __name__=="__main__":
    main()
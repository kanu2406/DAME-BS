import numpy as np
import pandas as pd
import sys,os,random,time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from kent.kent import kent_mean_estimator
from dame_bs.dame_bs import dame_with_binary_search
from experiments.real_data_experiments.globem.preprocess import *

def main():

    """
    Executes the mean estimation experiment on segmented step count data.

    Steps:
    -------
        - Load 4 CSV files containing Fitbit-derived step counts over different time segments of the day.
        - Preprocess the data: merge, remove NaNs, flatten per-user data, and truncate to uniform length.
        - Scale data to the [-1, 1] range and compute true means.
        - Shuffle users and run both estimation algorithms (Kent's and DAME-BS) 500 times.
        - Runs both algorithms 500 times to estimate the mean and records time and accuracy.
        - Reports runtime, mean estimates, and median MSE (in both scaled and original ranges) including 10th and 90th percentile.
        - Saves each trial along with other parameter in a csv file.
        - Also saves the final summary of results.
    """

    # Loading Data
    feature_dir = "experiments/Datasets/GLOBEM_Steps"
    files = [os.path.join(feature_dir, 'steps_sample_1.csv'),os.path.join(feature_dir, 'steps_sample_2.csv'),
            os.path.join(feature_dir, 'steps_sample_3.csv'),os.path.join(feature_dir, 'steps_sample_4.csv')]
    dfs = []
    
    for path in files:
        dfs.append(load_and_select_steps(path))
    # concatinating data from different files into one dataframe
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.sort_values(['pid','date']).reset_index(drop=True)
    print("Loaded Data.")
    print(df_all.shape)
    print(df_all.head())

    # count NaNs per column
    nan_per_col = df_all.isna().sum()
    print(nan_per_col)

    # count NaNs per row
    n_nan_rows = df_all.isna().any(axis=1).sum()
    print("Rows with ≥1 NaN:", n_nan_rows)


    # each user (pid) has a step count in four segments 
    seg_cols = ['steps_morning','steps_afternoon','steps_evening','steps_night']
    # Group by pid and collect all segment values into a single list per pid
    pid_and_steps = {}
    for pid, group in df_all.groupby('pid'):
        # extract the numpy array of shape (num_days, 4)
        arr = group[seg_cols].values
        # flatten to 1D and convert to Python list
        flat = arr.flatten().tolist()
        pid_and_steps[pid] = flat

    # Removing Nan values from data
    pid_and_steps_new = {}
    for pid, vals in pid_and_steps.items():
        pid_and_steps_new[pid] = [v for v in vals if not pd.isna(v)]

    # Truncating samples so that each user has same number of samples
    pid_and_steps_final,desired_length = truncating_data(pid_and_steps_new)
    print("Truncation Done.")

    # Scaling data in the range [-1,1]
    user_samples_scaled, vmin,vmax,true_mean,true_mean_scaled = scaling(pid_and_steps_final)
    print("Scaling Done.")

    print("Total number of users", len(user_samples_scaled))
    print("Number of samples per user : ",desired_length)
    print(f"True mean : {true_mean:.3f}" )
    print(f"True mean scaled : {true_mean_scaled:.3f}" )

    random.shuffle(user_samples_scaled)
    
    print("Running both algorithms 500 times")
    n = len(user_samples_scaled)
    m = desired_length
    alpha = 0.6
   

    num_exp = 500
    num_exp = 500
    base_seed = 42             
    out_trials_csv = "experiments/real_data_experiments/globem/results/steps_trials.csv"
    out_summary_csv = "experiments/real_data_experiments/globem/results/steps_summary.csv"
    

    # initialize output CSV
    init_results_csv(out_trials_csv)

    print(f"Running both algorithms for {num_exp} trials; writing to {out_trials_csv}")

    for trial in range(num_exp):
        seed = make_seed(base_seed, trial)
        np.random.seed(seed)
        random.seed(seed)
        counts=user_samples_scaled.copy()
        np.random.shuffle(counts)
        X=np.array(counts)

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

            # errors (scaled and original)
            scaled_mse_kent = (theta_hat_kent_scaled - true_mean_scaled) ** 2
            scaled_mse_dame = (theta_hat_dame_scaled - true_mean_scaled) ** 2
            orig_mse_kent = (theta_hat_kent_orig - true_mean) ** 2
            orig_mse_dame = (theta_hat_dame_orig - true_mean) ** 2

            status = "ok"

        except Exception as e:
            # on error, record NaNs and error message
            theta_hat_kent_scaled = np.nan
            theta_hat_dame_scaled = np.nan
            theta_hat_kent_orig = np.nan
            theta_hat_dame_orig = np.nan
            scaled_mse_kent = np.nan
            scaled_mse_dame = np.nan
            orig_mse_kent = np.nan
            orig_mse_dame = np.nan
            time_k = np.nan
            time_d = np.nan
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
            "theta_hat_kent_orig": float(theta_hat_kent_orig) if not np.isnan(theta_hat_kent_orig) else np.nan,
            "theta_hat_dame_orig": float(theta_hat_dame_orig) if not np.isnan(theta_hat_dame_orig) else np.nan,
            "scaled_mse_kent": float(scaled_mse_kent) if not np.isnan(scaled_mse_kent) else np.nan,
            "scaled_mse_dame": float(scaled_mse_dame) if not np.isnan(scaled_mse_dame) else np.nan,
            "orig_mse_kent": float(orig_mse_kent) if not np.isnan(orig_mse_kent) else np.nan,
            "orig_mse_dame": float(orig_mse_dame) if not np.isnan(orig_mse_dame) else np.nan,
            "time_kent_s": float(time_k) if not np.isnan(time_k) else np.nan,
            "time_dame_s": float(time_d) if not np.isnan(time_d) else np.nan,
            "status": status
        }
        append_row_csv(out_trials_csv, row)

    print(f"Trials complete. Aggregating results to {out_summary_csv}")

    # Aggregate and save summary 
    df = pd.read_csv(out_trials_csv)

    # compute summary
    summary = {
        "dataset": "GLOBEM_steps_count_data",
        "n": n,
        "m": m,
        "alpha": alpha,
        "trials": num_exp,
        "median_scaled_mse_kent": float(df["scaled_mse_kent"].median(skipna=True)),
        "median_scaled_mse_dame": float(df["scaled_mse_dame"].median(skipna=True)),
        "10pct_scaled_mse_kent": float(df["scaled_mse_kent"].quantile(0.1)),
        "90pct_scaled_mse_kent": float(df["scaled_mse_kent"].quantile(0.9)),
        "10pct_scaled_mse_dame": float(df["scaled_mse_dame"].quantile(0.1)),
        "90pct_scaled_mse_dame": float(df["scaled_mse_dame"].quantile(0.9)),
        "median_orig_mse_kent": float(df["orig_mse_kent"].median(skipna=True)),
        "median_orig_mse_dame": float(df["orig_mse_dame"].median(skipna=True)),
        "mean_time_kent_s": float(df["time_kent_s"].mean(skipna=True)),
        "mean_time_dame_s": float(df["time_dame_s"].mean(skipna=True)),
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
    print("-" * 60)
    print(" Mean Squared Error (Original values)")
    print(f"   Kent median MSE:   {summary['median_orig_mse_kent']:.4e}")
    print(f"   DAME median MSE:   {summary['median_orig_mse_dame']:.4e}")
    print("-" * 60)
    
    print(" Runtime (seconds)")
    print(f"   Kent mean runtime: {summary['mean_time_kent_s']:.4f}")
    print(f"   DAME mean runtime: {summary['mean_time_dame_s']:.4f}")
    print("-" * 60)
   

if __name__=="__main__":
    main()
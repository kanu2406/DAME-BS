import time
import numpy as np
import pandas as pd
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator
from experiments.real_data_experiments.stocks_data.preprocess import *
np.random.seed(42)


def main():
    """
    This script runs a 500-trial comparison of Kent's mean estimator and DAME-BS
    on real stock prives data from the yfinance.

    Steps:
        - Loads ticker symbols.
        - Download available close prices of 1000 stocks using loaded tickers over the period of 1 year for each day.
        - Computes the log returns and check for stationarity of time series.
        - Filter out all the time series with Nan values and truncate them so each of them have same number of sample.
        - Scaled the data in range [-1,1].
        - Runs both algorithms across 500 trials.
        - Reports runtime, mean estimates, and median MSE (in both scaled and original ranges) including 10th and 90th percentile.
        - Saves each trial along with other parameter in a csv file.
        - Also saves the final summary of results.

    """
    print("Starting...")
    # Fetching tickers symbols
    tickers = load_and_clean_tickers(n=2000)
    print("Loaded Tickers")
    # downloading close prices of each ticker
    # prices = batch_download_close_prices(tickers)
    # print("Downloaded Stock Prices")

    # save_dict(prices, "experiments/Datasets/stock_data.pkl") #saving the data
    prices = load_dict("experiments/Datasets/stock_data.pkl")


    # Computing log returns to have stationarity
    returns = compute_log_returns(prices)
    print("Computed log returns")
    # Checking stationarity
    stat = check_stationarity(returns)
    if stat==False:
        print("Since all seires are not stationary, results might not be good.")

    # Filtering series with Nan and truncating to have equal number of samples per  stock
    truncated,min_length = filter_and_truncate(returns, min_samples=249)
    print("Truncation Done.")
    # scaling all values to [-1,1]
    scaled_returns,true_mean, true_mean_scaled,min_val,max_val = scale_series(truncated)
    print("Scaling of returns in the range [-1,1] done.")


    print("Running both algorithms 500 times")
    n = len(scaled_returns)
    m = min_length
    alpha = 0.6
    

    num_exp = 500
    base_seed = 42             
    out_trials_csv = "experiments/real_data_experiments/stocks_data/results/stocks_trials.csv"
    out_summary_csv = "experiments/real_data_experiments/stocks_data/results/stocks_summary.csv"
    
    # initialize output CSV
    init_results_csv(out_trials_csv)

    print(f"Running both algorithms for {num_exp} trials; writing to {out_trials_csv}")

    for trial in range(num_exp):
        seed = make_seed(base_seed, trial)
        np.random.seed(seed)
        random.seed(seed)
        scaled=scaled_returns.copy()
        np.random.shuffle(scaled)
        X=np.array(scaled)

        try:
            # Kent
            t0 = time.time()
            theta_hat_kent_scaled = kent_mean_estimator(X, alpha=alpha, K=1.0)
            t1 = time.time()
            time_k = t1 - t0
            theta_hat_kent_orig = 0.5 * (theta_hat_kent_scaled + 1) * (max_val - min_val) + min_val

            # DAME-BS
            t0 = time.time()
            theta_hat_dame_scaled = dame_with_binary_search(n, alpha, m, X)
            t1 = time.time()
            time_d = t1 - t0
            theta_hat_dame_orig = 0.5 * (theta_hat_dame_scaled + 1) * (max_val - min_val) + min_val

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

    # ---- Aggregate and save summary ----
    df = pd.read_csv(out_trials_csv)

    # compute robust summaries (ignores NaNs)
    summary = {
        "dataset": "NASDAQ_returns",
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
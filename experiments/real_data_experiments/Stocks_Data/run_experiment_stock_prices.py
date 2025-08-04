import time
import numpy as np
import pandas as pd
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from experiments.real_data_experiments.stocks_data.preprocess import load_and_clean_tickers,check_stationarity
from experiments.real_data_experiments.stocks_data.preprocess import batch_download_close_prices,compute_log_returns
from experiments.real_data_experiments.stocks_data.preprocess import filter_and_truncate,scale_series
from experiments.real_data_experiments.stocks_data.preprocess import save_dict, load_dict
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator
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
        - Reports runtime, mean estimates, and MSE (in both scaled and original ranges).

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
    X=np.array(scaled_returns)

    num_exp = 500
    errors_kent = []
    errors_dame_bs = []
    true_err_kent = []
    true_err_dame_bs = []
    time_kent = []
    time_dame_bs = []
    theta_hats_kent = []
    theta_hats_dame_bs = []
    ests_kent = []
    ests_dame_bs = []

    for _ in range(num_exp):
        
        start_time_kent = time.time()
        theta_hat_kent   = kent_mean_estimator(X, alpha=0.6, K=1.0)
        time_kent.append(time.time() - start_time_kent)
        est_kent = 0.5 * (theta_hat_kent + 1) * (max_val - min_val) + min_val #estimated mean in the orignal range
        theta_hats_kent.append(theta_hat_kent)
        ests_kent.append(est_kent)

        start_time_dame_bs = time.time()
        theta_hat_dame_bs   = dame_with_binary_search(n, alpha, m, X)
        time_dame_bs.append(time.time() - start_time_dame_bs)
        est_dame_bs = 0.5 * (theta_hat_dame_bs + 1) * (max_val - min_val) + min_val #estimated mean in the orignal range
        theta_hats_dame_bs.append(theta_hat_dame_bs)
        ests_dame_bs.append(est_dame_bs)

        errors_kent.append( (theta_hat_kent - true_mean_scaled)**2 )
        errors_dame_bs.append( (theta_hat_dame_bs - true_mean_scaled)**2 )

        true_err_kent.append( (est_kent - true_mean)**2 )
        true_err_dame_bs.append( (est_dame_bs - true_mean)**2 )


     
    print("--------------------------STATS FOR KENT'S Algo --------------------------------")
    print(f"Time taken: {np.mean(time_kent):.5f}s")
    print(f"Estimated Mean in the range [-1,1]= {np.mean(theta_hats_kent):.4f}, true mean scaled = {true_mean_scaled:.4f}")
    print(f"Estimated mean = {np.mean(ests_kent):.6f}, true mean = {true_mean:.6f}")
    print(f"MSE between values in range [-1,1] (scaled) = {np.mean(errors_kent):.6f}")
    print(f"MSE (in original range) = {np.mean(true_err_kent):.6f}")
    print("--------------------------------------------------------------------------------")


    print("----------------------------STATS FOR DAME-BS ----------------------------------")
    print(f"Time taken: {np.mean(time_dame_bs):.5f}s")
    print(f"Estimated Mean in the range [-1,1]= {np.mean(theta_hats_dame_bs):.4f}, true mean scaled = {true_mean_scaled:.4f}")
    print(f"Estimated mean = {np.mean(ests_dame_bs):.6f}, true mean = {true_mean:.6f}")
    print(f"MSE between values in range [-1,1] (scaled) = {np.mean(errors_dame_bs):.6f}")
    print(f"MSE (in original range) = {np.mean(true_err_dame_bs):.6f}")
    print("--------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()

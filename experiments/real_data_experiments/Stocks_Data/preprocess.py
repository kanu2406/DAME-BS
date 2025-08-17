import pandas as pd
import numpy as np
import random
import yfinance as yf
import pickle
import time, os
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import hashlib



def load_and_clean_tickers(n=4996):
    """
    Loads and filters NASDAQ-listed stock tickers.

    Parameters
    ----------
    n : int
        Number of tickers to return after shuffling.

    Returns
    -------
    list of str
        A list of clean ticker symbols ready for data download.
    """
    

    # Loading from the official NASDAQ-listed tickers (includes symbols, ETF flags, etc.)
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    nasdaq_df = pd.read_csv(nasdaq_url, sep="|")

    print("Columns in NASDAQ listing:", nasdaq_df.columns.tolist())
    print("Total tickers available:", len(nasdaq_df))

    # Ensure Symbol column is string type and drop NaNs
    nasdaq_df['Symbol'] = nasdaq_df['Symbol'].astype(str)

    # Filter out test issues and special symbols safely
    cleaned_tickers = nasdaq_df[
        (nasdaq_df['Test Issue'] == 'N') &
        (~nasdaq_df['Symbol'].astype(str).str.contains(r'[$]'))
    ]['Symbol'].tolist()

    random.shuffle(cleaned_tickers)
    tickers = cleaned_tickers[:n]
    print(f"Using {len(tickers)} tickers.")

    return tickers


def batch_download_close_prices(tickers, period="1y", interval="1d",batch_size=100):
    """This function downloads stock prices in batches using tikcers over period 
    of 1 year with and interval of 1 day. It returns a dictionary with available 
    close prices of stocks over a year present in the tickers list. 

    Parameters
    ----------
    tickers : list of str
        List of stock symbols to fetch.
    period : str
        Time window for data retrieval (default is '1y').
    interval : str
        Interval between observations (default is '1d').
    batch_size : int
        Number of tickers per download batch.

    Returns
    -------
    dict
        Dictionary mapping tickers to their daily close price lists.
    """

    
    result = {}
    num_batches = np.ceil(len(tickers) / batch_size)
    for i in range(int(num_batches)):
        batch = tickers[i*batch_size : (i+1)*batch_size]
        df = yf.download(batch, period=period, interval=interval, progress=True,auto_adjust=True)
        for t in tqdm(batch):
            # We try to download available daily close prices of a stock
            try:
                prices = df['Close'][t].dropna().tolist() 
                if prices:
                    result[t] = prices
            except KeyError:
                continue
        time.sleep(2) # pausing between batches
    return result

def compute_log_returns(prices_dict):
    """
    Computes log returns from stock price series.

    Parameters
    ----------
    prices_dict : dict
        Dictionary mapping tickers to their price time series.

    Returns
    -------
    dict
        Dictionary mapping tickers to their log return time series.
    """
    returns = {
        t: np.diff(np.log(p)).tolist()
        for t, p in prices_dict.items()
    }
    return returns


def check_stationarity(returns_dict):
    """
    Checks stationarity of return series using Augmented Dickey-Fuller test.

    Parameters
    ----------
    returns_dict : dict
        Dictionary of stocks mapped to their log return time series.

    Returns
    -------
    bool
        True if all series are stationary (p-value < 0.05), False otherwise.
    """
    for t in returns_dict.keys():
        series = np.array(returns_dict[t])
        stat=True
        if len(series)>1:
            adf_result = adfuller(series)
            print(f"ADF Statistic: {adf_result[0]:.4f}")
            print(f"ADF p-value: {adf_result[1]:.4f}")
            if adf_result[1]>0.05:
                print(f"Stock prices corresponding to ticker {t} is not stationary.")
                stat= False
        if stat ==True:
            print("All given time series are stationary.")
        return stat
    
def filter_and_truncate(returns_dict, min_samples=230):
    """
    Filters out time series with NaNs or insufficient length,
    and truncates all remaining series to the minimum length.

    Parameters
    ----------
    returns_dict : dict
        Dictionary of log return time series.
    min_samples : int
        Minimum required number of samples for each time series.

    Returns
    -------
    tuple
        - dict: Truncated time series of equal length.
        - int: The truncation length used.
    """
    
    clean_dict = {ticker: prices
    for ticker, prices in returns_dict.items()
    if prices and not any(pd.isna(prices)) and len(prices) >= min_samples
    }

    print(f"Kept {len(clean_dict)} tickers with no NaNs,non-empty price lists and more than {min_samples} samples.")

    min_length = min(len(prices) for prices in clean_dict.values())

    # Truncating all price lists to min_length
    truncated_dict = {
        ticker: prices[:min_length]
        for ticker, prices in clean_dict.items()
    }

    print(f"All tickers truncated to {min_length} samples.")
    return truncated_dict,min_length

def scale_series(series_dict):
    """
    Scales all time series to the range [-1, 1] using min-max normalization.

    Parameters
    ----------
    series_dict : dict
        Dictionary of unscaled time series.

    Returns
    -------
    tuple
        - scaled_returns : list of list 
                Scaled and shuffled time series.
        - true_mean : float 
                Mean of original values.
        - true_mean_scaled : float 
                Scaled mean.
        - min_val : float 
                Minimum value in the original data.
        - max_val : float 
                Maximum value in the original data.
    """
    
    all_vals = [v for values in series_dict.values() for v in values]
    min_val = min(all_vals)
    max_val = max(all_vals)
    true_mean=np.mean(all_vals)
    scaled_returns=[]
    for pid, vals in series_dict.items():
            vals=np.array(vals)
            scaled = (2 * (vals - min_val) / (max_val - min_val)) - 1
            random.shuffle(scaled)
            scaled_returns.append(scaled.tolist())
    true_mean_scaled = (2 * (true_mean - min_val) / (max_val - min_val)) - 1
    return scaled_returns,true_mean, true_mean_scaled,min_val,max_val


def save_dict(d, path):
    """
    Saves a Python dictionary to a pickle file.

    Parameters
    ----------
    d : dict
        The dictionary to save.
    path : str
        File path to save the dictionary.
    """
    with open(path, "wb") as f:
        pickle.dump(d, f)

def load_dict(path):
    """
    Loads a Python dictionary from a pickle file.

    Parameters
    ----------
    path : str
        File path to load the dictionary from.

    Returns
    -------
    dict
        The loaded dictionary object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    



def make_file(path):
    """
    Ensure that the directory for a given file path exists.

    This function extracts the directory portion of the given file path 
    and checks if it exists. If it does not exist, it creates the 
    directory (including any intermediate directories as needed).

    Parameters
    ----------
    path : str
        A path-like object representing a file system path.

    Returns
    -------
    None
        This function does not return anything. It ensures the 
        parent directory exists.
    """
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def make_seed(base_seed, trial_index):
    """
    Generate a deterministic 32-bit integer seed from a base seed 
    and a trial index.

    The function concatenates the `base_seed` and `trial_index`, 
    hashes the string using MD5, and returns the first 32 bits 
    (8 hex characters) of the hash as an integer.

    Parameters
    ----------
    base_seed : int
        The base seed value used for reproducibility.
    trial_index : int
        The trial index used to generate unique seeds across trials.

    Returns
    -------
    int
        A deterministic 32-bit integer seed.
    """
    s = f"{int(base_seed)}_{int(trial_index)}"
    h = hashlib.md5(s.encode("utf8")).hexdigest()
    return int(h[:8], 16)  # 32-bit integer

def init_results_csv(path):
    """
    Initialize (create or overwrite) a results CSV file with 
    predefined column headers.

    This function ensures the parent directory exists, then 
    creates a CSV file with the expected schema for storing 
    per-trial results.

    Parameters
    ----------
    path : str
        File path where the CSV will be created.

    Returns
    -------
    None
        This function writes the CSV file to disk.
    """
    make_file(path)
    columns = [
        "trial",
        "seed",
        "n", "m", "alpha",
        "theta_hat_kent_scaled", "theta_hat_dame_scaled",
        "theta_hat_kent_orig",  "theta_hat_dame_orig",
        "scaled_mse_kent", "scaled_mse_dame",
        "orig_mse_kent", "orig_mse_dame",
        "time_kent_s", "time_dame_s",
        "status"
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)

def append_row_csv(path, row: dict):
    """
    Append a single row of results to an existing CSV file.

    This function ensures the parent directory exists and then 
    appends a new row (provided as a dictionary) to the CSV file. 
    If the file does not already exist, it is the caller's 
    responsibility to initialize it with `init_results_csv`.

    Parameters
    ----------
    path : str
        File path to the CSV file.
    row : dict
        A dictionary where keys correspond to column names and 
        values correspond to row entries.

    Returns
    -------
    None
        This function writes the new row to the CSV file.
    """
    make_file(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)








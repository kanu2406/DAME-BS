import pandas as pd
import numpy as np
import random
import yfinance as yf
import pickle
import time
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox



def load_and_clean_tickers(n=4996):
    """This function fetches and cleans the dataframe of 4998 tickers from nasdaq url
     and returns a clean list containing Symbols of tickers required to extract data from yfinance. """
    

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
    close prices of stocks over a year present in the tickers list. """

    
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
    """Returns log of returns of stock prices."""
    returns = {
        t: np.diff(np.log(p)).tolist()
        for t, p in prices_dict.items()
    }
    return returns


def check_stationarity(returns_dict):
    """Checks stationarity of time series of returns of stocks"""
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
    """Keep the time series with no Nans and having enough samples ."""
    
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
    """Scale all values of series to the range [-1,1]."""
    
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
    with open(path, "wb") as f:
        pickle.dump(d, f)

def load_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    











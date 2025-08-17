import random
import numpy as np
from collections import defaultdict
import pandas as pd
import os,hashlib


def scaling_data(user_samples):
    """
    Scales user data into the range [-1, 1] and computes relevant statistics.

    This function flattens all user data into a single list, finds the global
    minimum and maximum values, and then linearly scales all values for each user
    into the range [-1, 1] using the transformation:

        scaled_value = (2 * (value - vmin) / (vmax - vmin)) - 1

    Parameters
    ----------
    user_samples : dict
        A dictionary mapping users (keys) to lists of numeric samples (values),

    Returns
    -------
    user_samples_scaled : dict
        Dictionary with the same keys as `user_samples`, where each user's values
        are scaled to the range [-1, 1].

    desired_length : int
        The minimum number of samples across all users.

    true_mean : float
        The mean of all unscaled sample values.

    true_mean_scaled : float
        The mean of all scaled sample values.

    vmin : float
        Minimum value among all unscaled sample values.

    vmax : float
        Maximum value among all unscaled sample values.

    """

    all_vals = [v for vals in user_samples.values() for v in vals]
    true_mean = np.mean(all_vals)
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    user_samples_scaled = {
        uid: [(2*(v - vmin) / (vmax - vmin)) - 1 for v in vals]
        for uid, vals in user_samples.items()
    }
    uid_counts = { uid: len(vals)
               for uid, vals in user_samples_scaled.items() }
    desired_length=min(uid_counts.values())
    print("Number of samples per user : ",desired_length)
    all_vals_scaled = [v for vals in user_samples_scaled.values() for v in vals]
    true_mean_scaled = np.mean(all_vals_scaled)
    return user_samples_scaled,desired_length,true_mean,true_mean_scaled,vmin,vmax

def truncate_and_shuffle(user_samples,desired_length):
    
    """
    Truncates each user's sample list to a fixed length and shuffles it randomly.

    Parameters
    ----------
    user_samples : dict
        Dictionary mapping users to lists of samples.

    desired_length : int
        The number of samples to retain per user. 

    Returns
    -------
    final : dict
        Dictionary containing users with `desired_length` samples,
        where each user's samples are truncated and shuffled.

    """

    final = {}
    for uid, vals in user_samples.items():
        if len(vals) >= desired_length:
            vals_copy = vals.copy()
            random.shuffle(vals_copy)
            final[uid] = vals_copy[:desired_length]
    return final





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








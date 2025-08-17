import numpy as np
import pandas as pd
import random,os,hashlib

def load_and_select_steps(path):
    """
    Load and preprocess step count data from a CSV file containing steps data of users.
    This function reads the Fitbit intraday step data and selects only relevant columns: 
    participant ID (`pid`), date, and the sum of steps for each time segment of the day
    (morning, afternoon, evening, night). It then renames those columns for readability.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the step data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns: 'pid', 'date', 'steps_morning', 
        'steps_afternoon', 'steps_evening', and 'steps_night'.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    # picking the columns that end with one of our segments
    segments = ['morning', 'afternoon', 'evening', 'night']
    seg_cols = [c for c in df.columns
                if any(c.endswith(f"f_steps:fitbit_steps_intraday_rapids_sumsteps:{s}") for s in segments)]

    # keep pid, date, + those columns
    df = df[['pid','date'] + seg_cols]

    # rename each sum‑col to something like 'steps_morning'
    rename_map = {
        c: f"steps_{c.split(':')[-1]}"
        for c in seg_cols
    }
    return df.rename(columns=rename_map)


def truncating_data(data):
    """
    Truncates each user's data to the same number of samples (minimum length among all users),
    after shuffling their individual sample lists.

    Parameters
    ----------
    data : dict
        Dictionary where keys are user IDs and values are lists of numeric samples.

    Returns
    -------
    
        - dict: Truncated dictionary of user samples with uniform length.
        - int: Number of samples per user after truncation.
    """

    # Counting minimum number of samples for all users
    pid_counts = { pid: len(vals) for pid, vals in data.items() }
    desired_length=min(pid_counts.values())
    print("Each user will have a sample length of : ",desired_length)
    data_final = {}
    for pid, vals in data.items():
        if len(vals) >= desired_length:
            vals_copy = vals.copy()
            np.random.shuffle(vals_copy)
            data_final[pid] = vals_copy[:desired_length]

    return data_final,desired_length


def scaling(data):
    """
    Scales user sample data to the range [-1, 1] and computes the mean before and after scaling.

    Parameters
    ----------
    data : dict
        Dictionary where keys are user IDs and values are lists of numeric samples.

    Returns
    -------
    
        - user_samples_scaled : list of list 
                Scaled user samples.
        - min_val : float 
                Minimum value before scaling.
        - max_val : float 
                Maximum value before scaling.
        - true_mean : float 
                True mean before scaling.
        - true_mean_scaled : float
                True mean after scaling.
    """
    
    user_samples_scaled = []
    all_vals = [v for values in data.values() for v in values]
    min_val = min(all_vals)
    max_val = max(all_vals)
    true_mean=np.mean(all_vals)
    print("Mean: ",true_mean)
    for pid, vals in data.items():
        vals=np.array(vals)
        scaled = (2 * (vals - min_val) / (max_val - min_val)) - 1
        user_samples_scaled.append(scaled.tolist())
    
    all_vals_scaled = [v for vals in user_samples_scaled for v in vals]
    true_mean_scaled = np.mean(all_vals_scaled)

    return user_samples_scaled, min_val,max_val,true_mean,true_mean_scaled


def load_and_select_sleep(path):
    """
    Load and preprocess sleep duration data from a CSV file.
    This function reads Fitbit sleep data and selects only the participant ID (`pid`),
    date, and total duration of sleep (main sleep) for the day.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the sleep data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns: 'pid', 'date', and 'sleep_duration'.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['pid','date','f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:allday']]
    rename_map = {
        'f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:allday': 'sleep_duration'
    }
    return df.rename(columns=rename_map)





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








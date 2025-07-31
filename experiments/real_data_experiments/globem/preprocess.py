import numpy as np
import pandas as pd
import random

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
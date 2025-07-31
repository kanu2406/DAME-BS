import random
import numpy as np
from collections import defaultdict


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


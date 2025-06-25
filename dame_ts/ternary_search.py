import numpy as np
import math
import warnings
# from dame_ts.utils import min_n_required


def attempting_insertion_using_ternary_search(alpha,delta,n,m,user_samples):
    """
    Perform the localization phase of DAME using a private ternary search.

    This procedure partitions the interval [-1, 1] into three segments at each iteration,
    uses randomized response to privately count how many user sample-means fall into the
    left and right thirds, and discards the less-likely segment. It repeats until all groups
    are exhausted, returning the remaining interval.

    Parameters:
    -----------
    alpha : float
        Differential privacy parameter for the binary (randomized response) queries.
    delta : float
        Failure probability tolerated in the localization phase.
    n : int
        Number of users allocated to localization (should be n/2 of the total).
    m : int
        Number of samples per user.
    user_samples : list of array-like
        A list of length n, each entry an array-like of m real-valued samples for one user.

    Returns:
    --------
    List :
        [L,R] = The estimated interval that might contain true mean where L is the  lower and R is the upper bound.
    """
    # --- Input validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if n % 2 != 0:
        warnings.warn(f"n = {n} is odd; reducing n and user samples to {n - 1} to make it even.")
        n -= 1
        user_samples = user_samples[:n]
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(delta, (int, float)) or delta<=0 or delta>1:
        raise ValueError("delta must be a positive number less than 1")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if not isinstance(user_samples, (list, tuple)) or len(user_samples) != n:
        raise ValueError(f"user_samples must be a list of length {n}")

    # Check that each sample set is an iterable of length m
    for i, sample in enumerate(user_samples):
        if not hasattr(sample, '__len__') or len(sample) != m:
            raise ValueError(f"Each user sample must be an array-like of length {m}")
        
    # check n is greater than min_n_required
    # if n< min_n_required(alpha):
    #     warnings.warn(f"n = {n} is below the recommended minimum {min_n_required(alpha)}; result may be unreliable.")

    # Precomputing the probability of truthful response under randomized response
    if alpha == np.inf:
        pi_alpha=1
    else:
        pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))

    # Computing group size b_max to balance accuracy and privacy
    denom = np.log(2 * m / math.log(12)) - 2 * math.log(3 / 2)
    b1 = (2 * n * np.log(3 / 2)) / denom
    b2 = (2 / (((2 * pi_alpha - 1) ** 2) + 1e-3)) * np.log(2 * n / (delta +1e-3))
    b_max = int(math.ceil(max(b1, b2)))

    # Maximum number of ternary-search rounds
    t_max = n // b_max

    # Initializing search interval [L, R]
    L, R = -1.0, 1.0
    p = pi_alpha 

    # Performing t_max iterations of private ternary search
    for t in range(t_max):
        # Length of each third
        gamma = (R - L) / 3.0
        I1_L, I1_R = L, L + gamma  # left interval
        I3_L, I3_R = R - gamma, R  # right interval

        # Initialize noisy counts for left and right intervals
        V1_tilde = 0
        V3_tilde = 0

        # Process each user in the current group of size b_max
        start = t * b_max
        end = start + b_max
        for x in user_samples[start:end]:
            x_bar = np.mean(x)  # sample-mean for this user

            # Determine true membership in I1 and I3
            V1 = 1 if (I1_L <= x_bar <= I1_R) else 0
            V3 = 1 if (I3_L <= x_bar <= I3_R) else 0

            # Apply randomized response: truth with prob p, flip with prob 1-p
            if np.random.rand() < p:
                V1_tilde += V1
                V3_tilde += V3
            else:
                V1_tilde += 1 - V1
                V3_tilde += 1 - V3

        # Discard the interval (I1 or I3) with smaller noisy count
        if V1_tilde < V3_tilde:
            # Drop the left third: shift L to the start of I2
            L += gamma
        else:
            # Drop the right third: shift R to the end of I2
            R -= gamma

    # Final estimate of the interval containing true mean
    return [L,R]


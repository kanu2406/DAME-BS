import numpy as np
import math
from dame_ts.ternary_search import attempting_insertion_using_ternary_search
import warnings

def dame_with_ternary_search(n, alpha, m, user_samples):
    """
    Implements DAME algorithm with ternary search localization and Laplace estimation.

    Args:
        n: number of users (integer, even)
        alpha: privacy parameter
        m: number of samples per user
        user_samples: list or array of shape (n, m)
    Returns:
        bar_theta: aggregated estimator
    """

    # --- Input validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if n % 2 != 0:
        warnings.warn(f"n = {n} is odd; reducing it to {n - 1} to make it even.")
        n -= 1
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if not isinstance(user_samples, (list, tuple)) or len(user_samples) != n:
        raise ValueError(f"user_samples must be a list of length {n}")

    for i, sample in enumerate(user_samples):
        if not hasattr(sample, '__len__') or len(sample) != m:
            raise ValueError(f"Each user sample must be an array-like of length {m}")


    # Compute tau and delta
    tau = (2 * math.log(max(8 * math.sqrt(m * n) * (alpha ** 2), 1))) / m
    if alpha==np.inf:
        pi_alpha=1
    else:
        pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    # Localization phase
    # use first half of users for localization
    X1 = user_samples[:int(n/2)]
    

    # loc_samples = user_samples[:n//2]
    theta_hat_loc = attempting_insertion_using_ternary_search(alpha, delta, n//2, m, X1)

    # Estimation phase using second half
    X2 = user_samples[int(n/2):]
    # est_samples = user_samples[n//2:]
    
    hat_thetas = []
    scale = 14 * tau / alpha
    for x in X2:
        x_bar = np.mean(x)
        if x_bar < theta_hat_loc[0]:
            x_bar = theta_hat_loc[0]
        if x_bar > theta_hat_loc[1]:
            x_bar = theta_hat_loc[1]
        noisy = x_bar + scale * np.random.laplace(0, 1)
        noisy = max(-1, min(1, noisy))
        hat_thetas.append(noisy)

    # Aggregation
    bar_theta = (2 / n) * sum(hat_thetas)
    return bar_theta




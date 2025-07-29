import numpy as np
import math
from dame_bs.binary_search import attempting_insertion_using_binary_search
import warnings
from scipy.special import lambertw

def dame_with_binary_search(n, alpha, m, user_samples):
    """
    Implements DAME algorithm with Binary search (localization phase) and Estimation Phase.

    Parameters:
    -----------
        n : int (even)
          number of users 
        alpha : float
          privacy parameter
        m : int 
          number of samples per user
        user_samples: list or array of shape (n, m)
          User samples that will be used to estimate the mean.

    Returns:
    --------
        bar_theta : float 
          estimated mean
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
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if not isinstance(user_samples, (list, tuple)) or len(user_samples) != n:
        raise ValueError(f"user_samples must be a list of length {n}")

    for i, sample in enumerate(user_samples):
        if not hasattr(sample, '__len__') or len(sample) != m:
            raise ValueError(f"Each user sample must be an array-like of length {m}")
    
    if isinstance(user_samples, np.ndarray):
        # fast path for arrays
        if not np.all((user_samples >= -1) & (user_samples <= 1)):
            raise ValueError("All entries must lie in [-1, 1]")
    else:
        # if it is a list
        for i, row in enumerate(user_samples):
            # converting each row to array for convenience
            arr = np.asarray(row)
            if arr.size == 0:
                continue
            if arr.min() < -1 or arr.max() > 1:
                raise ValueError(f"All entries must lie in [-1, 1].Entry at index {i} contains values outside [-1,1]")
        
    # check m is greater than or equal to 7
    if m< 7:
        warnings.warn(f"m = {m} is below the recommended minimum 7; result may be unreliable.")


    
    if alpha==np.inf:
        pi_alpha=1
    else:
        pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    
    # Initializing parameters
    if alpha == np.inf:
        # overall = np.mean([np.mean(x) for x in user_samples])
        # return float(np.clip(overall, -1, 1))
        delta = max(2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2),1e-5)
        delta_prime = 0
        scale = 0
    else:
    
        delta_prime = np.sqrt((1 / m) * lambertw((32 * alpha**2 * n * m) / 81).real)


        term1 = 2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)
        logA = np.log(81 / (8 * alpha**2))
        logB = np.log(n)
        logC = np.log(81 / (8 * n * alpha**2))
        term2_inside_sqrt = logA**2 - 4 * logB * logC + 2 * n * (2 * pi_alpha - 1)**2 * np.log(3/2)
        term2 = np.exp(0.5 * logA - 0.5 * np.sqrt(term2_inside_sqrt))
        delta = min(max(term1, term2),1)


        
        inner_log_A = np.log(np.sqrt((9 * np.log(12)) / (8 * m)))
        floor_A = np.floor(inner_log_A / np.log(2/3))
        termA = (2/3)**floor_A
        numerator_B = n * (2 * pi_alpha - 1)**2
        denominator_B = 2 * np.log(2 * n / delta)
        floor_B = np.floor(numerator_B / denominator_B)
        termB = (2/3)**floor_B
        max_term = max(termA, termB)
        scale = (2 / alpha) * max_term + (2 * delta_prime / alpha)


    # Localization phase
    # use first half of users for localization
    X1 = user_samples[:int(n/2)]

    [L,R] = attempting_insertion_using_binary_search(alpha, delta, n//2, m, X1)
    L_tilde=max(L-delta_prime,-1)
    R_tilde=min(R+delta_prime,1)

    # Estimation phase using second half
    X2 = user_samples[int(n/2):]
    
    hat_thetas = []
    
    for x in X2:
        x_bar = np.mean(x)
        if x_bar < L_tilde:
            x_bar = L_tilde
        if x_bar > R_tilde:
            x_bar = R_tilde
        noisy = x_bar + scale * np.random.laplace(0, 1)
        noisy = max(-1, min(1, noisy))
        hat_thetas.append(noisy)

    # Aggregation
    bar_theta = (2 / n) * sum(hat_thetas)
    return bar_theta




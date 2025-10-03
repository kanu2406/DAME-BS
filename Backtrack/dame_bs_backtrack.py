import numpy as np
import math
from Backtrack.bs_backtrack import attempting_insertion_using_bs_backtrack
import warnings
from scipy.special import lambertw

def dame_with_binary_search_backtrack(n, alpha, m, user_samples):
   

    # --- Input validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")

    if user_samples.shape != (n, m):
        raise ValueError(f"user_samples must be a 2D array of shape ({n}, {m})")
    if n % 2 != 0:
        warnings.warn(f"n = {n} is odd; reducing n and user samples to {n - 1} to make it even.")
        n -= 1
        user_samples = user_samples[:n]

    if not np.all((user_samples >= -1) & (user_samples <= 1)):
        raise ValueError("All entries in user_samples must lie in [-1, 1]")

    # check m is greater than or equal to 7
    if m< 7:
        warnings.warn(f"m = {m} is below the recommended minimum 7; result may be unreliable.")


    
    if alpha==np.inf:
        pi_alpha=1
    else:
        pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    
    # Initializing parameters
    if alpha == np.inf:
        
        delta = max(2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2),1e-5)
        delta_prime = 0
        scale = 0
    else:
    
        delta_prime = np.sqrt((1 / m) * lambertw((16 * alpha**2 * n * m) / 9).real)


        term1 = 2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)
        logA = np.log(81 / (8 * alpha**2))
        logB = np.log(n)
        logC = np.log(81 / (8 * n * alpha**2))
        term2_inside_sqrt = logA**2 - 4 * logB * logC + 2 * n * (2 * pi_alpha - 1)**2 * np.log(3/2)
        term2 = np.exp(0.5 * logA - 0.5 * np.sqrt(term2_inside_sqrt))
        delta = min(max(term1, term2),1)
        delta=1/n


        
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
    
    [L,R] = attempting_insertion_using_bs_backtrack(alpha, delta, int(n/2), m, X1)
    L_tilde=max(L-delta_prime,-1)
    R_tilde=min(R+delta_prime,1)

    # Estimation phase using second half
    X2 = user_samples[int(n/2):]

    x_bars = np.mean(X2, axis=1)
    x_bars_clipped = np.clip(x_bars, L_tilde, R_tilde)
    scale = R_tilde-L_tilde
    noises = np.random.laplace(0, scale, size=x_bars_clipped.shape)
    noisy_estimates = np.clip(x_bars_clipped + noises, -1, 1)
    
    
    # Aggregation
    bar_theta = (2 / n) * sum(noisy_estimates)
    return bar_theta




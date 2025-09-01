import numpy as np
import math
import warnings,sys,os
from scipy.linalg import hadamard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from girgis.scalar import *
import numpy as np
import math
import warnings
from scipy.linalg import hadamard





def next_power_of_two(x):
    """
    Compute the smallest power of two greater than or equal to x.

    Parameters:
    -----------
        x : int
            Input integer (must be >= 1).

    Returns:
    --------
        int
            The least power of two such that result >= x.
            - If x <= 1, returns 1.

    """
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())





def mean_vector(D, tau, eps0, B=1.0, gamma=0.05):
    """
    Differentially private estimation of a high-dimensional mean vector
    using randomized Hadamard transform, range estimation, and 
    per-coordinate privatized user means.

    
    Parameters:
    -----------
        D : np.ndarray, shape (n, m, d)
            Input dataset with:
            - n : number of users
            - m : samples per user
            - d : dimension (must be a power of two)
            All entries must lie within [-B, B].
        tau : float
            Concentration radius parameter 
        eps0 : float
            Total user-level privacy budget.
        B : float, optional (default=1.0)
            Known absolute bound on each coordinate of the input samples.
        gamma : float, optional (default=0.05)
            Small failure probability (δ-like parameter) used in the 
            computation of tau_prime. Must be > 0.

    Returns:
    --------
        x_hat : np.ndarray, shape (d,)
            Differentially private estimate of the d-dimensional mean vector.
    
    """
    

    # validate shapes
    if D.ndim != 3:
        raise ValueError("D should have shape (n, m, d)")
    n, m, d = D.shape
    if not is_power_of_two(d):
        raise ValueError("d must be power of 2 for Hadamard transform")
    if not np.all((D >= -B) & (D <= B)):
        raise ValueError("All entries of D must lie in [-B,B]")

    # sample w in {+1,-1}^d and compute U = (1/sqrt(d)) H_d diag(w)
    w = np.random.choice([-1.0, 1.0], size=d)
    Hd = hadamard(d).astype(float)
    Hd_norm = Hd / np.sqrt(d)       
    U = Hd_norm @ np.diag(w)        

    

    # compute eps0' and tau'
    eps0_prime = eps0 / (2.0 * d)   # budget per coordinate for Range
    # Compute tau' = 10 * tau * sqrt(log(n d / delta)) / d
    if gamma <= 0:
        raise ValueError("delta must be > 0")
    tau_prime = 10.0 * tau * np.sqrt(np.log(( n * d / gamma)) / float(d))

    # adjusting upper bound after hadamard transform
    B_y = B * math.sqrt(d)
    

    # Adjusting tau_prime so B/tau_prime is integer & power of two (required by range_scalar)
    raw_k = max(1, int(round(B_y / tau_prime)))
    k_pow2 = next_power_of_two(raw_k)
    # recompute tau_prime consistent with chosen k
    tau_prime = B_y / float(k_pow2)

    
    Y = np.zeros_like(D, dtype=float)
    for i in range(n):
        for j in range(m):
            Y[i, j, :] = U @ D[i, j, :]

    # For each coordinate l, build Dl (n, m) using Y[..., l], then call range_scalar with eps0_prime and tau_prime
    ranges = [None] * d
    
    
    for l in range(d):
        Dl = Y[:, :, l]   # shape (n,m)
        Rl = range_scalar(Dl, tau_prime, eps0_prime, B=B_y)
        ranges[l] = Rl
        

   
    
    eps_mean = eps0 / 2.0   # half budget for the mean reporting step
    Z_reports = np.zeros((n, d), dtype=float)
    for i in range(n):
        j = np.random.randint(0, d)  # sampled coordinate
        [a_j, b_j] = ranges[j]
        # user's samples in coordinate j
        user_coord_samples = Y[i, :, j]   # length m
        # privatize scalar mean for this coordinate
        z_scalar = mean_user_scalar(user_coord_samples, a_j, b_j, eps_mean,tau_prime)

       
        zi = np.zeros(d, dtype=float)
        zi[j] = d * z_scalar
        Z_reports[i] = zi

   
    z_hat = Z_reports.mean(axis=0)   # length d
    x_hat = U.T @ z_hat
    # x_hat = np.clip(x_hat, -B, B)

    
    return x_hat

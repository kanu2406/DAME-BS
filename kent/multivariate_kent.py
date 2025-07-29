import numpy as np
from scipy.stats import laplace
from kent.kent import partition_interval
import warnings

def kent_multivariate_estimator(X, alpha, K=1.0):
    """
    Implements the Kent's algorithm for mean estimation under user level LDP when 
    the data is in l_inf ball of radius 1 i.e. all coordinates of the vector are in [-1,1].
    
    Parameters
    ----------
        X : array, shape (n, T, d) 
            User Samples in [-1,1]^d
        alpha : float
            Privacy Parameter
        K : float (>=0)
            Constant (Default = 1).
            
    Returns
    --------
        theta_hat : float
            Final d-dimensional estimated mean
    """
    

    # --- Input validation ---
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError("X must be a 3D numpy array of shape (n, T, d)")
    n, T, d = X.shape

    if n < (2*d):
        warnings.warn(
            f"n = {n} is less than 2*d = {2*d}. Not enough data for localization or estimation of all coordinates; returning zero vector."
        )
        return np.zeros(d)
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    # check n is a multiple of 2d
    if n%(2*d) != 0:
        n_new=n-(n%(2*d))
        X = X[:n_new]
        n = n_new
        warnings.warn(f"n = {n} was not a multiple of 2d = {2*d}. Adjusting number of users to the nearest lower multiple of 2d = {n_new}.")
    if not isinstance(K, (int, float)) or K <= 0:
        raise ValueError("K must be a positive number")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if not isinstance(X, (list, tuple,np.ndarray)) or len(X) != n:
        raise ValueError(f"user_samples must be a list of length {n}")
    if not np.all((X >= -1) & (X <= 1)):
        raise ValueError("All entries of X must lie in [-1, 1]")
    


    for i, sample in enumerate(X):
        if not hasattr(sample, '__len__') or len(sample) != T:
            raise ValueError(f"Each user sample must be an array-like of length {T}")

    if alpha == np.inf: 
        mu = X.mean(axis=(0, 1))  # shape (d,)
        # clip to [-1,1] in each coordinate
        return np.clip(mu, -1, 1)

    # to prevent overflow, checking if arg is very high
    LOG_MAX = np.log(np.finfo(float).max)  
    arg = n * alpha**2 / (K * d)
    arg_safe=min(LOG_MAX,arg)
    T_star = min(T, int(np.exp(arg_safe)) + 1)
    


    delta = np.sqrt(2 * np.log(n * T_star * alpha**2 / d) / T_star)
    
    half = n//2
    block = n//(2*d) # number of users per coordinate per phase
    
    # Partition [-1, 1] into intervals of length ~2*delta
    intervals = partition_interval(-1.0, 1.0, delta)
   
    # Dictionaries to store the estimated interval for each coordinate
    k_star = {} 
    Ik_bounds = {}

    for j in range(d):
        idx_loc = slice(j*block, (j+1)*block)      # first half of the block for localization
        
        V = np.zeros((len(X[idx_loc, :T_star, j]), len(intervals)), dtype=int)
        for i in range(len(X[idx_loc, :T_star, j])):

            # jth coordinate of empirical mean of T_star samples of ith user of the block
            theta_hat_i_j = np.mean(X[idx_loc, :T_star, j][i]) 
           
            for k in range(len(intervals)):
                # union of neighbor intervals
                neigh = intervals[max(k-1,0):min(k+2,len(intervals))]
                if any(L_i <= theta_hat_i_j < U_i for (L_i, U_i) in neigh):
                    V[i,k] = 1
        
        p = np.exp(alpha/6)/(1 + np.exp(alpha/6))
        U_rand = np.random.rand(len(X[idx_loc, :T_star, j]), len(intervals))
        Ve = np.where(U_rand <= p, V, 1 - V)
        
        k_star[j] = np.argmax(Ve.sum(axis=0)) # index of interval with the most votes
        Lj, Uj = intervals[k_star[j]]
        Lj_tilde = Lj - 6 * delta
        Uj_tilde = Uj + 6 * delta
        Ik_bounds[j] = (Lj_tilde, Uj_tilde)
    

    theta_hat_final = np.zeros(d)
    for j in range(d):
        Lj_tilde, Uj_tilde = Ik_bounds[j]
        idx_est = slice(half + j*block, half + (j+1)*block)  # second half for estimation

        theta_hat_j_refined = []

        for i in range(len(X[idx_est, :T_star, j])):
            theta_i_j = np.mean(X[idx_est, :T_star, j][i])
            
            # Project onto extended interval and add Laplace noise
            theta_proj = np.clip(theta_i_j, Lj_tilde, Uj_tilde)
            noise = (14 * delta / alpha) * laplace.rvs()
            theta_hat_i_j = theta_proj + noise
            theta_hat_i_j = max(-1, min(1, theta_hat_i_j))
            theta_hat_j_refined.append(theta_hat_i_j)

        theta_hat_final[j] = np.mean(theta_hat_j_refined)

    return theta_hat_final
    

import numpy as np
from kent.kent import kent_mean_estimator
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

    half = n//2
    block = n//(2*d) # number of users per coordinate per phase
    
    theta_hat = np.zeros(d)
    for j in range(d):
        idx_loc = slice(j * block, (j + 1) * block)
        idx_est = slice(half + j * block, half + (j + 1) * block)
        idx_loc_arr = np.arange(*idx_loc.indices(X.shape[0]))
        idx_est_arr = np.arange(*idx_est.indices(X.shape[0]))
        union_indices = np.concatenate([idx_loc_arr, idx_est_arr])
        X_union_j = X[union_indices, :, j]
        # Running Univariate Kent for each coordinate
        theta_hat[j]=kent_mean_estimator(X_union_j, alpha, K=1.0)


    # return theta_hat_final
    return theta_hat
    

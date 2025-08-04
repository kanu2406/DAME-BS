import numpy as np
from scipy.special import lambertw
from scipy.stats import laplace
from dame_bs.binary_search import attempting_insertion_using_binary_search
import warnings
import numpy as np
from scipy.special import lambertw
from scipy.stats import laplace
from dame_bs.binary_search import attempting_insertion_using_binary_search
import warnings
from dame_bs.dame_bs import dame_with_binary_search

def multivariate_dame_bs_l_inf(user_samples, alpha):
    """
    Multivariate mean estimation under LDP for support in l_inf ball 
    of radius 1 i.e. all coordinates of the vector are in [-1,1].

    Parameters:
    -----------
        user_samples : array, shape (n, m, d), values in [-1,1]^d
            User samples that will be used to estimate the mean.
        alpha : float
            Privacy Parameter
        m     : int
            Number of samples per user

    Returns:
    --------
        theta_hat : array, shape (d,), 
            Estimated d-dimensional mean 
    """

    if not isinstance(user_samples, np.ndarray) or user_samples.ndim != 3:
        raise ValueError("user_samples must be a 3D numpy array of shape (n, m, d)")

    n, m, d = user_samples.shape
    
    # --- Input validation ---
    if n < (2*d):
        warnings.warn(
            f"n = {n} is less than 2*d = {2*d}. Not enough data for localization or estimation of all coordinates; returning zero vector."
        )
        return np.zeros(d)
    
    # check n is a multiple of 2d
    if n%(2*d) != 0:
        # n_new = ((n + 2*d - 1) // (2*d)) * (2*d)
        n_new=n-(n%(2*d))
        user_samples = user_samples[:n_new]
        n = n_new
        warnings.warn(f"n = {n} was not a multiple of 2d = {2*d}. Adjusting number of users to the nearest lower multiple of 2d = {n_new}.")
    if not np.all((user_samples >= -1) & (user_samples <= 1)):
        raise ValueError("All entries must lie in [-1, 1]")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
        
    # check m is greater than or equal to 7
    if m< 7:
        warnings.warn(f"m = {m} is below the recommended minimum 7; result may be unreliable.")

    half = n//2
    block = n//(2*d) # number of users per coordinate per phase

    theta_hat = np.zeros(d)
    for j in range(d):

        # assigning n/d users for each coordinate 
        idx_loc = slice(j * block, (j + 1) * block)
        idx_est = slice(half + j * block, half + (j + 1) * block)
        idx_loc_arr = np.arange(*idx_loc.indices(user_samples.shape[0]))
        idx_est_arr = np.arange(*idx_est.indices(user_samples.shape[0]))
        union_indices = np.concatenate([idx_loc_arr, idx_est_arr])
        X_union_j = user_samples[union_indices, :, j]
        
        #Running algorithm for each coordinate
        theta_hat[j]=dame_with_binary_search(int(n/d), alpha, m, X_union_j)


    return theta_hat

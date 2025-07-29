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
    
    

    if alpha==np.inf:
        pi_alpha=1
    else:
        pi_alpha=np.exp(alpha)/(1+np.exp(alpha))
    
    if alpha == np.inf:
        delta_prime = np.sqrt((1/m) * lambertw(32*alpha*alpha*n*m/(81*d)).real)
        delta = max(2 * (n/d) * np.exp(-(n/d) * (2 * pi_alpha - 1)**2 / 2),1e-7)
        scale = 0
        
    
    else:
        delta_prime = np.sqrt((1/m) * lambertw(32*alpha*alpha*n*m/(81*d)).real)


        term1 = 2 * (n/d) * np.exp(-(n/d) * (2 * pi_alpha - 1)**2 / 2)
        logA = np.log(81 / (8 * alpha**2))
        logB = np.log(n/d)
        logC = np.log((81*d) / (8 * n * alpha**2))
        term2_inside_sqrt = logA**2 - 4 * logB * logC + 2 * n * (2 * pi_alpha - 1)**2 * np.log(3/2)
        term2 = np.exp(0.5 * logA - 0.5 * np.sqrt(term2_inside_sqrt))
        delta = min(max(term1, term2),1)



        inner_log_A = np.log(np.sqrt((9 * np.log(12)) / (8 * m)))
        floor_A = np.floor(inner_log_A / np.log(2/3))
        termA = (2/3)**floor_A
        numerator_B = n * (2 * pi_alpha - 1)**2
        denominator_B = 2 * np.log(2 * n / (d*delta))*d
        floor_B = np.floor(numerator_B / denominator_B)
        termB = (2/3)**floor_B
        max_term = max(termA, termB)
        scale = (2 / alpha) * max_term + (2 * delta_prime / alpha)




    theta_hat = np.zeros(d)
    half = n//2
    block = n//(2*d) # number of users per coordinate per phase

    for j in range(d):

        # indices for coordinate j
        idx_loc = slice(j*block, (j+1)*block)        # first half 
        idx_est = slice(half + j*block, half + (j+1)*block)  # second half 

        X_loc_j = user_samples[idx_loc, :, j]  # samples for localization phase
        L_j, R_j = attempting_insertion_using_binary_search(alpha, delta,len(user_samples[idx_loc]),m, X_loc_j.tolist())

        # expanding the interval by δ'
        L_tilde, R_tilde = L_j - delta_prime, R_j + delta_prime
        X_est_j = user_samples[idx_est, :, j]  # samples for estimation phase
        means = X_est_j.mean(axis=1)        # per-user mean

        # clipping into [L_tilde, R_tilde]
        clipped = np.minimum(np.maximum(means, L_tilde), R_tilde)
        # adding Laplace noise with scale = scale
        noisy = clipped + laplace.rvs(scale=scale, size=block)
        # aggregation to get coordinate estimate
        theta_hat[j] = (2*d/n) * np.sum(noisy)

    return theta_hat

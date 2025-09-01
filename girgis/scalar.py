import numpy as np
import math
import warnings
from scipy.linalg import hadamard

def mean_user_scalar(user_samples, a, b, eps0, tau):

    """
    Privately release a clipped per-user mean using Laplace noise and final projection.
    
    Parameters:
    -----------
        user_samples : array-like
            One user's samples (1D array or sequence). 
        a : float
            Lower bound for clipping and final projection. 
        b : float
            Upper bound for clipping and final projection. 
        eps0 : float
            Privacy parameter 
        tau : float
            used for scaling laplacian noise. Scale used for Laplacian Noise is 12*tau/eps0

    Returns:
    --------
        z_proj : float
            The privatized mean lying [a, b].

    
    """
    
    
    if not (a <= b):
        raise ValueError("Require a <= b")
    y = float(np.mean(user_samples))
    # clip to [a,b]
    y_clipped = float(np.minimum(np.maximum(y, a), b))
    span = 12*tau
    laplace_scale = span / eps0
    z = y_clipped + np.random.laplace(0.0, laplace_scale)
    
    # project back onto [a,b]
    z_proj = float(np.minimum(np.maximum(z, a), b))
    return z_proj



def is_power_of_two(k):
    """
    Check whether an integer k is a power of two.

    Parameters:
    -----------
        k : int
            The integer to test.

    Returns:
    --------
        bool
            True if k is a power of two (1, 2, 4, 8, ...),
            False otherwise.
    """
    return k > 0 and (k & (k - 1)) == 0





def rangeuser_scalar(D, tau, eps0, T):
    """
    Implements the Hadamard-based LDP mechanism for range localization.

    Parameters:
    -----------
        D : np.ndarray, shape (m,)
            User's samples, each in [-B, B].
        tau : float
            Concentration radius (not used explicitly in this step, but relevant for T construction).
        eps0 : float
            User-level LDP parameter.
        T : np.ndarray, shape (k,)
            Set of middle points of the intervals (the centers).

    Returns:
    --------
        z : float
            User's privatized response 
    """

    m_samples = len(D)
    # user's mean
    y = np.mean(D)   

    # nearest center index
    nu = int(np.argmin(np.abs(T - y)))  # index in [0, k-1]

    k = len(T) # total number of centers

    # Hadamard matrix of size k
    # k must be a power of 2 
    if (k & (k - 1)) != 0:
        raise ValueError("k must be a power of 2 for Hadamard matrix construction")
    Hk = hadamard(k).astype(float)

    # construct vector m = (1/sqrt(k)) * Hk * e_nu
    e_nu = np.zeros(k)
    e_nu[nu] = 1.0
    m = (Hk @ e_nu)/ np.sqrt(k) 

    # sample j ~ Unif[k]
    j = np.random.randint(0, k)

    # Probability parameters
    exp_eps = np.exp(eps0)
    coeff = (exp_eps + 1) / (exp_eps - 1)
    pj = 0.5 + 0.5 * np.sqrt(k) * m[j] * (exp_eps - 1) / (exp_eps + 1)

    
    if np.random.rand() < pj:
        z = +Hk[j, :] * coeff
    else:
        z = -Hk[j, :] * coeff

    return z




def range_scalar(D, tau, eps0, B=1.0):
    """
    Distributed Private Range Estimation for Scalars.

    Parameters:
    -----------
        D : np.ndarray of shape (n, m)
            Dataset, n users each with m samples in [-B, B].
        tau : float
            Concentration radius.
        epsilon0 : float
            User-level LDP parameter.
        B : float
            Bound on data values (data ∈ [-B, B]).
        
    Returns:
    --------
        R : List of floats
            Estimated range [amax - 3*tau, amax + 3*tau].
    """
    n, m = D.shape
    if not np.all((D >= -B) & (D <= B)):
        raise ValueError("All user samples must lie within [-B, B]")

    # Construct intervals of width 2*tau across [-B, B]
    if tau <= 0:
        raise ValueError("tau must be positive")
    
    k = B / tau
    if not k.is_integer():
        raise ValueError(f"B/tau = {B}/{tau} = {B/tau} is not an integer")
    k = int(k)

    # check k is a power of 2
    if not is_power_of_two(k):
        raise ValueError(f"k = {k} must be a power of 2 for Hadamard mechanism")
    
    edges = np.linspace(-B, B, k + 1)
    centers = (edges[:-1] + edges[1:]) / 2  # bin midpoints

    Z = []
    for i in range(n):
        zi = rangeuser_scalar(D[i], tau, eps0, centers)
        Z.append(zi)
    Z = np.vstack(Z)  # shape (n, k)

    # Server aggregates
    z_avg = Z.mean(axis=0)

    # Find argmax
    amax = centers[np.argmax(z_avg)]

    # Final range
    R = [amax - 3 * tau, amax + 3 * tau]
    return R

    



def meanscalar(D, tau, eps0, B=1.0):
    """
    Privately estimate the mean of a 2D dataset (user x samples) 
    using range estimation and per-user privatized means.
    It first estimate the global value range [a, b] of the dataset `D`
    by calling `range_scalar` with half the privacy budget (eps0/2).
    For each user i:
       - Compute the average of their samples D[i, :].
       - Call `mean_user_scalar` with [a, b] and the privacy budget.
       - This clips, noised, and projects the user mean.
    Then, aggregate the privatized user-level means across all users by taking their arithmetic mean.

    Parameters:
    -----------
        D : np.ndarray, shape (n, m)
            Dataset with `n` users, each contributing `m` scalar samples.
        tau : float
            Concentration radius
        eps0 : float
            Total privacy budget for the whole algorithm. 
            Internally split between range estimation (eps0/2) 
            and mean privatization (eps0/2).
        B : float, optional (default=1.0)
            Known absolute bound on all samples. All entries of `D`
            must lie in [-B, B].

    Returns:
    --------
        x_hat : float
            The privatized global mean estimate for the dataset.

    """
    
    n, m = D.shape

    # Compute private range using half the privacy budget
    [a,b] = range_scalar(D, tau, eps0/2,B)
    

    # Each user computes privatized mean in [a, b] with eps0/2
    z_list = [mean_user_scalar(D[i], a, b, eps0/2, tau) for i in range(n)]
    
    # Aggregation
    x_hat = np.mean(z_list)
    # x_hat = np.clip(x_hat,-B,B)
    return x_hat












import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.utils import plot_errorbars
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator

def generate_univariate_scaled_data(distribution,n,m, true_mean):

    """
    Generate and linearly scale univariate user samples into [-1,1].
    This function simulates `n` users each providing `m` samples drawn from a
    specified one‑dimensional distribution centered (in expectation) at
    `true_mean`, and then rescales *all* samples (and the true mean) to lie in
    the interval [-1, 1].

    Parameters
    ----------
    distribution : str
        Which distribution to sample from. Supported values are:
        
        - "normal"    : Gaussian N(true_mean, 1^2)
        - "uniform"   : Uniform U(true_mean - 1, true_mean + 1)
        - "student_t" : Student's t distribution with degrees of freedom = 3 
                        and expected value shifted to true_mean
        - "binomial"  : Binomial with number of trials as 50 and probability of success as true_mean/50

    n : int
        Number of users (i.e., how many independent sample‐sets to generate).
    m : int
        Number of i.i.d. samples per user.
    true_mean : float
        The ground‐truth mean around which samples are generated; also used
        to compute the scaled “true_mean” output.

    Returns
    -------
    user_samples_scaled : ndarray of shape (n, m)
        The generated samples for all users, after rescaling linearly so the
        *minimum* across all samples maps to -1 and the *maximum* maps to +1.
    true_mean_scaled : float
        The location of `true_mean` on the same linear map (i.e. the point in
        [-1,1] where the original `true_mean` falls).

    """

    if distribution=="normal":
        # Generate user samples: n users × m samples, sampled from N(true_mean, 1)
        user_samples = [np.random.normal(loc=true_mean, scale=1, size=m) for _ in range(n)]
    if distribution=="uniform":
        # Uniform
        user_samples = [np.random.uniform(low=true_mean-1, high=true_mean+1, size=m) for _ in range(n)]
    if distribution=="standard_t":
        # standard_t
        user_samples = np.random.standard_t(df=3, size=(n, m)) + true_mean
    if distribution=="binomial":
        # Binomial 
        user_samples = np.random.binomial(n=50,p=true_mean/50,size=(n, m)).astype(float)


    
    vmin = np.min(user_samples)
    vmax = np.max(user_samples)
    eps=1e-5
    rng = vmax - vmin

    # if all draws are identical
    if rng==0:
        print("vmax and vmin are equal. Mapping everything to zero.")
        user_samples_scaled = np.zeros_like(user_samples)
        true_mean_scaled = 0.0

    else:
        safe_rng = np.where(rng < eps, 1.0, rng)
        # scaling in [-1,1]^d
        user_samples_scaled = (2 * (user_samples - vmin) / safe_rng) - 1  
        true_mean_scaled   = (2 * (true_mean   - vmin) / safe_rng) - 1

    return user_samples_scaled, true_mean_scaled

def compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=50):
    """
    Compare DAME‑BS and Kent univariate estimators over multiple trials.

    In each trial, this function:
      1. Generates `n` users × `m` samples from the specified `distribution`
         centered (in expectation) at `true_mean`.
      2. Scales all samples and `true_mean` into [-1,1].
      3. Runs both the DAME‑BS estimator and the Kent estimator under privacy
         budget `alpha`.
      4. Computes the squared error of each estimate vs. the scaled `true_mean`.

    Parameters
    ----------
    n : int
        Number of users.
    m : int
        Number of i.i.d. samples per user.
    alpha : float
        Local differential privacy parameter.
    distribution : str
        Which distribution to sample from. Supported Distributions:
          - "normal"
          - "uniform"
          - "student_t"
          - "binomial"
    true_mean : float
        Ground-truth mean of the underlying distribution.
    trials : int, optional
        Number of independent trials to average over (default: 50).

    Returns
    -------
    median_err_dame : float
        Median of squared errors from the DAME-BS estimator across trials.
    lower_err_dame : float
        First decile of  squared errors from the DAME-BS estimator across trials.
    upper_err_dame : float
        Last decile of squared errors from the DAME-BS estimator across trials.
    median_err_kent : float
        Median of squared errors from the Kent estimator across trials.
    lower_err_kent : float
        First decile of squared errors from the Kent estimator across trials.
    upper_err_kent : float
        Last decile of squared errors from the Kent estimator across trials.

    """

    error_dame_bs = []
    error_kent = []
    for _ in range(trials):
        user_samples_scaled,true_mean_scaled = generate_univariate_scaled_data(distribution,n,m, true_mean)
        
        estimate = dame_with_binary_search(n, alpha, m, user_samples_scaled)
        err1 = (estimate - true_mean_scaled)**2
        error_dame_bs.append(err1)
        estimate = kent_mean_estimator(user_samples_scaled,alpha)
        err2 = (estimate - true_mean_scaled)**2
        error_kent.append(err2)
    # return np.mean(error_dame_bs),np.std(error_dame_bs),np.mean(error_kent),np.std(error_kent)
    return np.median(error_dame_bs),np.percentile(error_dame_bs, 10, axis=0),np.percentile(error_dame_bs, 90, axis=0),np.median(error_kent),np.percentile(error_kent, 10, axis=0),np.percentile(error_kent, 90, axis=0)
    




def experiment_risk_vs_param_for_dist_univariate(distribution,param_to_vary="alpha",param_values=None,
                                      n=8000,m=20,alpha=0.6,true_mean=0.3,trials=50):
    
    """
    For the specified parameter `param_to_vary` (α, n, or m), 
    plots MSE of two univariate LDP estimators by DAME-BS and Kent's algorithm.

    For each value in `param_values`, this function runs
    `compare_univariate_algorithms` to obtain mean±std of squared error
    for both DAME‑BS and Kent estimators, and then calls `plot_errorbars`.

    Parameters
    ----------
    distribution : str
        Supported Data distributions ("normal", "uniform", "student_t", "binomial").
    param_to_vary : str, optional
        Which parameter to vary: "alpha", "n", or "m" (default: "alpha").
    param_values : array-like, optional
        Values to use for specified parameter to vary. If None, defaults are:
          - alpha → np.linspace(0.1, 1.0, 20)
          - n     → range(500, 16000, 200)
          - m     → range(7, 500, 5)
    n : int, optional
        Base number of users (used when param_to_vary ≠ "n"; default: 8000).
    m : int, optional
        Base samples per user (used when param_to_vary ≠ "m"; default: 20).
    alpha : float, optional
        Base privacy parameter (used when param_to_vary ≠ "alpha"; default: 0.6).
    true_mean : float, optional
        Ground‑truth mean (default: 0.3).
    trials : int, optional
        Trials per setting (default: 50).

    Raises
    ------
    ValueError
        If `param_to_vary` is not one of "alpha", "n", "m".

    Returns
    -------
    None
        Displays a plot of MSE vs. the chosen parameter for both estimators.

    
    """
   
    if param_to_vary not in {"alpha", "n", "m"}:
        raise ValueError("param_to_vary must be one of 'alpha', 'n', or 'm'")

    if param_values is None:
        if param_to_vary == "alpha":
            param_values = np.linspace(0.1, 1.0, 20)
        elif param_to_vary == "n":
            param_values = list(range(500, 16000  , 200))
        elif param_to_vary == "m":
            param_values = list(range(7, 500, 5))

    median_errors_dame = []
    lower_errors_dame = []
    upper_errors_dame = []
    median_errors_kent = []
    lower_errors_kent = []
    upper_errors_kent = []

    
    if param_to_vary == "alpha":
       
        for alpha in param_values:
            # Running both algorithms
            print(f"Running algorithms for alpha = {alpha}")
            median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=trials)
            
            median_errors_dame.append(median_dame)
            lower_errors_dame.append(lower_dame)
            upper_errors_dame.append(upper_dame)
            median_errors_kent.append(median_kent)
            lower_errors_kent.append(lower_kent)
            upper_errors_kent.append(upper_kent)
    
    elif param_to_vary=="n":
        
        for n in param_values:
            # Running both algorithms
            print(f"Running algorithms for n = {n}")
            median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=trials)
            
            median_errors_dame.append(median_dame)
            lower_errors_dame.append(lower_dame)
            upper_errors_dame.append(upper_dame)
            median_errors_kent.append(median_kent)
            lower_errors_kent.append(lower_kent)
            upper_errors_kent.append(upper_kent)

    else:
        for m in param_values:
            print(f"Running algorithms for m = {m}")
            # Running both algorithms
            median_dame,lower_dame,upper_dame,median_kent,lower_kent,upper_kent= compare_univariate_algorithms(n,m,alpha,distribution,true_mean,trials=trials)
            
            median_errors_dame.append(median_dame)
            lower_errors_dame.append(lower_dame)
            upper_errors_dame.append(upper_dame)
            median_errors_kent.append(median_kent)
            lower_errors_kent.append(lower_kent)
            upper_errors_kent.append(upper_kent)


    # Labels
    xlabel_map = {
        "alpha": "Privacy parameter α",
        "n": "Number of users n",
        "m": "Samples per user m"
    }

    title = f"Mean Squared Error vs {xlabel_map[param_to_vary]} for {distribution} distribution"
   
    
    plot_errorbars(param_values, median_errors_kent,median_errors_dame, lower_errors_kent,
                   lower_errors_dame,upper_errors_kent, upper_errors_dame, xlabel_map[param_to_vary], 
                   ylabel="Mean Squared Error", title=title,log_scale=True,plot_ub=False,upper_bounds=None)
    




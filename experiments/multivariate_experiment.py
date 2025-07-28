import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.utils import plot_errorbars
from dame_bs.multivariate_dame_bs import multivariate_dame_bs_l_inf
from kent.multivariate_kent import kent_multivariate_estimator


def generate_multivariate_scaled_data(distribution, n, m,d, true_mean):
    """
    Generates and scales multivariate samples into [-1,1]^d.

    Parameters
    ----------
    distribution : str
        Supported Distributions {"normal","uniform","student_t","binomial"}.
    n : int
        Number of users (i.e. number of independent sample‐sets).
    m : int
        Number of samples per user.
    d : int
        Dimension of each dimension.
    true_mean : array‐like of shape (d,)
        The ground‐truth mean vector.

    Returns
    -------
    user_samples_scaled : ndarray, shape (n, m, d)
        All user samples, scaled so each coordinate lies in [-1,1].
    true_mean_scaled : ndarray, shape (d,)
        The true_mean vector under the same per‐coordinate scaling.
    """
    true_mean = np.asarray(true_mean, dtype=float)
    d = true_mean.shape[0]

    # Generating raw samples of shape (n, m, d)
    if distribution == "normal":
        user_samples = np.random.normal(loc=true_mean, scale=1.0, size=(n, m, d))
    elif distribution == "uniform":
        user_samples = np.random.uniform(low=true_mean - 1, high=true_mean + 1, size=(n, m, d))
    elif distribution == "student_t":
        user_samples = np.random.standard_t(df=3, size=(n, m, d)) + true_mean
    elif distribution == "binomial":
        user_samples = np.random.binomial(n=50,p=true_mean/50,size=(n, m,d)).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Compute per‐coordinate min and max across all users and samples
    vmin = user_samples.min(axis=(0, 1))  # shape (d,)
    vmax = user_samples.max(axis=(0, 1))  # shape (d,)
    eps=1e-5
    rng = vmax - vmin
    user_samples_scaled = np.zeros_like(user_samples)
    true_mean_scaled = np.zeros_like(true_mean)

    
    for k in range(d):
        if rng[k] == 0:
            # if all values of that coordinate are equal
            print(f"Generated values of the coordinate {d} are equal so mapping all of them to zero.")
            user_samples_scaled[:, :, k] = 0.0
            true_mean_scaled[k]      = 0.0
        else:
            # safe denominator
            safe_rng = rng[k] if rng[k] > eps else eps
            # scaling in [-1,1]^d
            user_samples_scaled[:, :, k] = (2 * (user_samples[:, :, k] - vmin[k]) / safe_rng) - 1
            true_mean_scaled[k]  = (2 * (true_mean[k]    - vmin[k]) / safe_rng )- 1


    return user_samples_scaled, true_mean_scaled






def compare_multivariate_algorithms(n,m,d,alpha,distribution,true_mean,trials=50):
    """
    Compare two multivariate LDP mean‐estimators (Kent vs. DAME‑BS) over multiple trials.

    For each of `trials` iterations, this function:
      1. Generates an (n × m × d) dataset from `distribution` with expected mean `true_mean`.
      2. Scales every coordinate into [-1, 1] (and scales `true_mean` accordingly).
      3. Runs the Kent multivariate estimator under privacy budget `alpha`.
      4. Runs the DAME‑BS multivariate estimator under the same budget.
      5. Computes the squared ℓ₂‐error of each estimate vs. the scaled true mean.

    Parameters
    ----------
    n : int
        Number of users 
    m : int
        Number of samples per user 
    d : int
        Dimension of each sample
    alpha : float
        Local differential privacy parameter.
    distribution : str
        Sampling distribution: one of {"normal", "uniform", "student_t", "binomial"}.
    true_mean : array‐like of shape (d,)
        The ground‐truth mean vector in R^d.
    trials : int, optional
        How many independent datasets (and runs) to average over (default: 50).

    Returns
    -------
    mean_err_kent : float
        Mean of squared ℓ₂‐errors from the Kent estimator across trials.
    std_err_kent : float
        Standard deviation of those errors.
    mean_err_dame : float
        Mean of squared ℓ₂‐errors from the DAME‑BS estimator across trials.
    std_err_dame : float
        Standard deviation of those errors.

    """
    true_mean = np.asarray(true_mean, dtype=float)
    errors_kent = []
    errors_dame = []

    for _ in range(trials):
        X_scaled, mu_scaled = generate_multivariate_scaled_data(distribution, n, m, d,true_mean)

        # Kent estimator
        theta_kent = kent_multivariate_estimator(X_scaled, alpha)
        err_kent   = np.linalg.norm(theta_kent - mu_scaled)**2
        errors_kent.append(err_kent)

        # DAME‑BS estimator
        theta_dame = multivariate_dame_bs_l_inf(X_scaled, alpha)
        err_dame   = np.linalg.norm(theta_dame - mu_scaled)**2
        errors_dame.append(err_dame)

    return np.mean(errors_kent),np.std(errors_kent),np.mean(errors_dame),np.std(errors_dame)
    


def experiment_risk_vs_param_for_dist_multivariate(distribution,param_to_vary="alpha",param_values=None,
                                      n=8000,m=20,d=10,alpha=0.6,true_mean=[0.1]*10,trials=50):
    """
    For the specified parameter `param_to_vary` (α, n, m or d), 
    plots ℓ₂‐errors of two univariate LDP estimators by DAME-BS and Kent's algorithm.

    For each value in `param_values`, this function runs
    `compare_multivariate_algorithms` to obtain mean±std of squared ℓ₂‐error
    for both DAME‑BS and Kent estimators, and then calls `plot_errorbars`.

    Parameters
    ----------
    distribution : str
        Data distribution ("normal", "uniform", "student_t", "binomial").
    param_to_vary : str, optional
        Which parameter to vary: "alpha", "n", or "m" (default: "alpha").
    param_values : array-like, optional
        Values to use for specified parameter to vary. If None, defaults are:
          - alpha -> np.linspace(0.1, 1.0, 20)
          - n     -> range(500, 16000, 200)
          - m     -> range(7, 500, 5)
          - d     -> list(range(2,500,10))
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
        If `param_to_vary` is not one of "alpha", "n", "m","d.

    Returns
    -------
    None
        Displays a plot of mean squared ℓ₂‐error vs. the chosen parameter for both estimators.

    
    """
    
    if param_to_vary not in {"alpha", "n", "m","d"}:
        raise ValueError("param_to_vary must be one of 'alpha', 'n', 'm' or 'd'")
    if len(true_mean)!=d:
        raise ValueError(f"True mean should have same dimension as the param d ={d} specified")

    if param_values is None:
        if param_to_vary == "alpha":
            param_values = np.linspace(0.1, 1.0, 20)
        elif param_to_vary == "n":
            param_values = list(range(500, 16000  , 200))
        elif param_to_vary == "m":
            param_values = list(range(7, 500, 5))
        elif param_to_vary == "d":
            param_values = list(range(2,500,10))

    mean_errors_dame = []
    std_errors_dame = []
    mean_errors_kent = []
    std_errors_kent = []

    if param_to_vary == "alpha":
        for alpha in param_values:
            # Running both algorithms
            mean_kent, std_kent, mean_dame, std_dame = compare_multivariate_algorithms(n,m,d,alpha,distribution,true_mean,trials=trials)

            mean_errors_kent.append(mean_kent)
            std_errors_kent.append(std_kent)
            mean_errors_dame.append(mean_dame)
            std_errors_dame.append(std_dame)
        
    elif param_to_vary=="n":
        for n in param_values:
            # Running both algorithms
            mean_kent, std_kent, mean_dame, std_dame = compare_multivariate_algorithms(n,m,d,alpha,distribution,true_mean,trials=trials)

            mean_errors_kent.append(mean_kent)
            std_errors_kent.append(std_kent)
            mean_errors_dame.append(mean_dame)
            std_errors_dame.append(std_dame)

    elif param_to_vary == "d":
        for d in param_values:
            # Running both algorithms
            true_mean = [0.1]*d # each time dimension changes, true mean's dimension will also change 
            mean_kent, std_kent, mean_dame, std_dame = compare_multivariate_algorithms(n,m,d,alpha,distribution,true_mean,trials=trials)

            mean_errors_kent.append(mean_kent)
            std_errors_kent.append(std_kent)
            mean_errors_dame.append(mean_dame)
            std_errors_dame.append(std_dame)
    
    elif param_to_vary=="m":
        for m in param_values:
            # Running both algorithms
            mean_kent, std_kent, mean_dame, std_dame = compare_multivariate_algorithms(n,m,d,alpha,distribution,true_mean,trials=trials)

            mean_errors_kent.append(mean_kent)
            std_errors_kent.append(std_kent)
            mean_errors_dame.append(mean_dame)
            std_errors_dame.append(std_dame)


    # Labels
    xlabel_map = {
        "alpha": "Privacy parameter α",
        "n": "Number of users n",
        "m": "Samples per user m",
        "d": "Dimension of each sample d"
    }

    title = f"Mean Squared Error vs {xlabel_map[param_to_vary]} for {distribution} distribution"

    
    plot_errorbars(param_values, mean_errors_kent,mean_errors_dame, std_errors_kent,
                   std_errors_dame, xlabel_map[param_to_vary], ylabel="Squared l_2 Error", title=title,
                   log_scale=True,plot_ub=False,upper_bounds=None)


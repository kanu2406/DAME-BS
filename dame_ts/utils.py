import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 
from dame_ts import dame_with_ternary_search


def min_n_required(alpha):
    '''
    Computes the minimum number of users (n) required to satisfy a privacy and accuracy condition 
    for the DAME-TS algorithm, based on the given differential privacy parameter α.

        n ≥ (4 / (2π_α - 1)^4) * (√2 + √(2 + ln(3/2)(2π_α - 1)^2))^2

    where π_α = e^α / (1 + e^α)

    Parameters
    ----------
    alpha : float
        The differential privacy parameter (α > 0). Higher α means less privacy.

    Returns
    -------
    float
        The minimum required number of users `n` to ensure the localization step satisfies 
        the theoretical guarantees. Returns `np.inf` if α is too small, making the denominator 
        near zero and the bound undefined.
    '''

    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    two_pi_minus_1 = 2 * pi_alpha - 1

    if abs(two_pi_minus_1) < 1e-6:
        return np.inf  # Avoid division by near-zero

    ln_32 = np.log(3 / 2)
    term1 = 4 / (two_pi_minus_1 ** 4)
    term2 = np.sqrt(2) + np.sqrt(2 + ln_32 * (two_pi_minus_1 ** 2))
    return term1 * (term2 ** 2)




def theoretical_upper_bound(alpha, n, m):
    '''
    Computes a theoretical upper bound on the mean squared error (MSE) of the DAME algorithm's 
    estimate of the true mean, under differential privacy constraints.

    Parameters
    ----------
    alpha : float
        Differential privacy parameter (α > 0). 
    n : int
        Total number of users.
    m : int
        Number of samples per user.

    Returns
    -------
    float
        A theoretical upper bound on the expected squared error:
        E[(μ - 𝜃̂)^2], where μ is the true mean and 𝜃̂ is the estimate returned by DAME.


    '''
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    term1 = (9 * np.log(12)) / (8 * m) + 8 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    denom = np.sqrt(2) + np.sqrt(2 + np.log(3/2) * (2 * pi_alpha - 1)**2)
    exponent = ((2 * pi_alpha - 1)**2) / denom * np.sqrt(n)
    term2 = (2/3)**exponent + 8 * n * np.exp(-np.sqrt(np.log(3/2) * (2 * pi_alpha - 1)**2 * n))

    return max(term1, term2)




def run_dame_experiment(n, alpha, m, true_mean, trials=50,distribution="normal"):
    """
    Runs the DAME algorithm multiple times to empirically evaluate its estimation error 
    under different data distributions and privacy settings.

    This function simulates a scenario with `n` users, each having `m` samples generated 
    from a specified distribution centered around a known `true_mean`. 

    Parameters
    ----------
    n : int
        Total number of users participating in the experiment.
    alpha : float
        Differential privacy parameter. Higher values allow more accurate estimation
        with weaker privacy guarantees.
    m : int
        Number of samples collected per user.
    true_mean : float
        The actual mean of the underlying data distribution. Used to compute estimation error.
    trials : int, optional
        Number of times the experiment is repeated (default is 50).
    distribution : str, optional
        Type of distribution to sample user data from. Supported values:
            - "normal"     : Samples from N(true_mean, 0.5^2)
            - "uniform"    : Samples from U(true_mean - 1, true_mean + 1)
            - "laplace"    : Samples from Laplace(true_mean, 0.5)
            - "exponential": Samples from Exponential(1.0) shifted to have mean ≈ true_mean

    Returns
    -------
    mean_error : float
        The average squared error between the DAME estimate and the true mean over all trials.
    std_error : float
        The standard deviation of the squared error over all trials.

    """


    errors = []
    for _ in range(trials):
        
        if distribution=="normal":
            # Generate user samples: n users × m samples, sampled from N(true_mean, 0.5^2)
            user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
        if distribution=="uniform":
            # Uniform
            user_samples = [np.random.uniform(low=true_mean-1, high=true_mean+1, size=m) for _ in range(n)]
        if distribution=="laplace":
            # Laplace
            user_samples = [np.random.laplace(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
        if distribution=="exponential":
            # Exponential shifted by true_mean - 1/λ so mean approx true_mean
            user_samples = [np.random.exponential(scale=1.0, size=m) + (true_mean - 1.0) for _ in range(n)]

            
        # Run algorithm
        estimate = dame_with_ternary_search(n, alpha, m, user_samples)
        
        # Measure error
        error = (estimate - true_mean)**2
        errors.append(error)
    
    return np.mean(errors), np.std(errors)


def plot_errorbars_and_upper_bounds(alphas, mean_errors, std_errors,upper_bounds, xlabel, ylabel, title):
    plt.figure(figsize=(8, 5))
    plt.errorbar(alphas, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.plot(alphas, upper_bounds, 'r--', label='Theoretical Upper Bound')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


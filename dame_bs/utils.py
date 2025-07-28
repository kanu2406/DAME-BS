import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 



def theoretical_upper_bound(alpha, n, m):
    '''
    Computes a theoretical upper bound on the mean squared error (MSE) of the algorithm's 
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
        E[(theta_hat - theta)^2], where theta is the true mean and theta_hat is the estimate returned by DAME-BS.
        The theoretical bound is given by -

        E[(theta_hat - theta)^2]<= 389/nmα^2 * {log(2nmα^2) v 1}  + 
                        16 max{   2nexp(-n*((2*pi_alpha-1)**2)/2),    (9/α√8) * exp(-(2*pi_alpha-1)*√n/4) }

    '''
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError("alpha must be a positive number")
    if alpha == np.inf:
        pi_alpha=1
    else:
        pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))

    if alpha == np.inf:
        return 16*2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    # First term: (389 / (nmα²)) * max{ln(2nmα²), 1}
    first_inner_log = np.log(2 * n * m * alpha**2)
    first_term = (389 / (n * m * alpha**2)) * max(first_inner_log, 1)

    # Second term: max{2n * exp(...), (9 / (α√8)) * exp(...)}
    exp_term1 = 2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    exp_term2 = (9 / (alpha * np.sqrt(8))) * np.exp(-0.25 * (2 * pi_alpha - 1) * np.sqrt(n))
    second_term = 16 * max(exp_term1, exp_term2)

    # Final result
    result = first_term + second_term
    return result



def plot_errorbars(x_values, mean_errors_kent,mean_errors_dame_bs, std_errors_kent,
                   std_errors_dame_bs, xlabel, ylabel, title,log_scale=True,plot_ub=False,upper_bounds=None):
    """
    Plots error bars for dame_bs and kent algorithms on a single graph.

    This function is typically used to visualize the mean squared errors for different values (e.g., alpha or n or m).

    Args:
        x_values (list or array-like): X-axis values (e.g., alpha values or user counts).
        mean_errors_kent (list or array-like): Mean squared error values corresponding to `alphas` for kent algorithm.
        mean_errors_dame_bs (list or array-like): Mean squared error values corresponding to `alphas` for dame_bs algorithm.
        std_errors_kent (list or array-like): Standard deviation of the errors for each value in `alphas` for kent algorithm.
        std_errors_dame_bs (list or array-like): Standard deviation of the errors for each value in `alphas` for dame_bs algorithm.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        plot_ub (Bool) : If true then plots theoretical upper bounds for dame_bs algorithm. Default is 'False'.
        upper_bounds (list or array-like): Theoretical upper bound values for dame_bs algorithm corresponding to `alphas'. Default is empty-list.

    Returns:
        None. Displays the plot using `matplotlib.pyplot`.
    """

    if upper_bounds is None:
        upper_bounds=[]
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, mean_errors_kent, yerr=std_errors_kent, fmt='o-', capsize=5,label="Kent")
    plt.errorbar(x_values, mean_errors_dame_bs, yerr=std_errors_dame_bs, fmt='o-', capsize=5,label="DAME-BS")
    if plot_ub and upper_bounds:
        plt.plot(x_values, upper_bounds, 'r--', label='Theoretical Upper Bound')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    plt.grid(True)
    plt.legend() 
    plt.show()















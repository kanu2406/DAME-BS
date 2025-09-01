import math
import csv,time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator
from girgis.scalar import *



def generate_univariate_scaled_data(distribution,n,m, true_mean,seed=42):

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
        - "standard_t" : Student's t distribution with degrees of freedom = 3 
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
    r = np.random.default_rng(seed)

    if distribution=="normal":
        # Generate user samples: n users × m samples, sampled from N(true_mean, 1)
        user_samples = [r.normal(loc=true_mean, scale=1, size=m) for _ in range(n)]
        user_samples = np.array(user_samples)
    if distribution=="uniform":
        # Uniform
        user_samples = [r.uniform(low=true_mean-1, high=true_mean+1, size=m) for _ in range(n)]
        user_samples = np.array(user_samples)
    if distribution=="standard_t":
        # standard_t
        user_samples = r.standard_t(df=3, size=(n, m)) + true_mean
    if distribution=="binomial":
        # Binomial 
        # user_samples = np.random.binomial(n=50,p=true_mean/50,size=(n, m)).astype(float)
        # user_samples = np.array(user_samples)
        # lam = 3.0 if true_mean is None else float(true_mean)
        # lambdas = np.full(n, lam, dtype=float)
        # user_samples = r.poisson(lam=lam, size=(n, m)).astype(float)
        # true_mean = float(lam)
        p = true_mean
        trials = 50
        user_samples = r.binomial(trials, p, size=(n, m)).astype(float)
        # print(user_samples)
        user_samples = user_samples / float(trials)   # now nominally in [0,1]
        actual_mean = trials*p
        true_mean = actual_mean / float(trials)


    
    
    if distribution == "binomial":
        vmin,vmax = 0,1
    else:
        vmin = np.min(user_samples)
        vmax = np.max(user_samples)
        # low_q, high_q = 0.05,0.99
        # vmin = float(np.quantile(flat, low_q))
        # vmax = float(np.quantile(flat, high_q))
        user_samples = (user_samples-vmin) / (vmax-vmin)   # now nominally in [0,1]
        true_mean = (true_mean-vmin) /(vmax-vmin)

    data_clipped = np.clip(user_samples, vmin, vmax)
    jitter_eps = 1e-6
    # if jitter_eps and jitter_eps > 0:
    noise = np.random.uniform(low=-jitter_eps, high=jitter_eps, size=data_clipped.shape)
    data_clipped = data_clipped + noise


    
    vmin = np.min(user_samples)
    vmax = np.max(user_samples)
    data_clipped = np.clip(user_samples, vmin, vmax)
    user_samples = data_clipped
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










def plot_errorbars(x_values, median_errors_kent,median_errors_dame_bs, lower_errors_kent,
                   lower_errors_dame_bs,upper_errors_kent, upper_errors_dame_bs,
                   median_errors_girgis,lower_errors_girgis,upper_errors_girgis, xlabel, 
                   ylabel, title,log_scale=True,plot_ub=False,upper_bounds=None,save_path=None,
                   log_log_scale = False,y_lim=True):
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
        upper_bounds (list or array-like): Theoretical upper bound values for dame_bs algorithm corresponding to `alphas`. Default is empty-list.
        save_path (str) : If save_path is provided then saves the generated plot to provided path else displays plot. Default is None.

    Returns:
        None. Displays the plot using `matplotlib.pyplot` or or saves the plot at the given path.
    """

    if upper_bounds is None:
        upper_bounds=[]
    plt.figure(figsize=(8, 5))
    plt.fill_between(x_values, lower_errors_kent, upper_errors_kent, alpha=0.3)
    plt.plot(x_values, median_errors_kent, label="Kent")
    
    plt.fill_between(x_values, lower_errors_dame_bs, upper_errors_dame_bs, alpha=0.3)
    plt.plot(x_values, median_errors_dame_bs,label="DAME-BS")
    
    plt.fill_between(x_values, lower_errors_girgis, upper_errors_girgis, alpha=0.3)
    plt.plot(x_values, median_errors_girgis,label="Girgis")
    
    all_lowers = np.minimum(np.minimum(lower_errors_kent, lower_errors_dame_bs), lower_errors_girgis)
    all_uppers = np.maximum(np.maximum(upper_errors_kent, upper_errors_dame_bs), upper_errors_girgis)

    y_min = np.min(all_lowers) * 0.05
    y_max = np.max(all_uppers) * 5.8
    y_min = max(y_min, 1e-8)

    if plot_ub and upper_bounds:
        plt.plot(x_values, upper_bounds, 'r--', label='Theoretical Upper Bound')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lim:
        plt.ylim(y_min, y_max)
    if log_scale & log_log_scale:
        print("Both log scale and log_log scale cannot")
    if log_scale:
        plt.yscale('log')
    if log_log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid(True)
    plt.legend() 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show()










def single_trial(n, m, alpha, distribution, true_mean, trial_seed):
    """
    Run one complete experimental trial.

    This function generates a fresh dataset with the given random seed,
    runs both the DAME-BS and Kent estimators on it, and records their
    performance metrics (MSE and runtime). The output is formatted as a
    dictionary suitable for logging to CSV.

    Parameters
    ----------
    n : int
        Number of users.
    m : int
        Number of samples per user.
    alpha : float
        Privacy parameter passed to both estimators.
    distribution : str
        Distribution type to generate synthetic data from
        (supported values : "normal","uniform","binomial","standard_t").
    true_mean : float
        Ground-truth mean of the distribution (before scaling).
    trial_seed : int
        Random seed controlling reproducibility of this trial.

    Returns
    -------
    row : dict
        Dictionary containing per-trial information with keys:
        - "n", "m", "alpha", "distribution", "true_mean", "seed"
        - "dame_estimate", "dame_mse", "dame_time",
        - "kent_estimate", "kent_mse", "kent_time",
        - "girgis_estimate","girgis_mse","girgis_time",
        - "status" : "ok" or error string
    """
    try:
        # Generate data (uses seed)
        user_samples_scaled, true_mean_scaled = generate_univariate_scaled_data(
            distribution=distribution, n=n, m=m, true_mean=true_mean, seed=int(trial_seed)
        )

        # run DAME-BS 
        t0 = time.time()
        est_dame = dame_with_binary_search(n, alpha, m, user_samples_scaled)
        t1 = time.time()
        dame_time = t1 - t0
        dame_mse = float((est_dame - true_mean_scaled) ** 2)

        # run Kent
        t0 = time.time()
        est_kent = kent_mean_estimator(user_samples_scaled, alpha)
        t1 = time.time()
        kent_time = t1 - t0
        kent_mse = float((est_kent - true_mean_scaled) ** 2)

        # run Girgis
        pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
        term1 = 2 * n * np.exp(-n * (2 * pi_alpha - 1)**2 / 2)
        logA = np.log(81 / (8 * alpha**2))
        logB = np.log(n)
        logC = np.log(81 / (8 * n * alpha**2))
        term2_inside_sqrt = logA**2 - 4 * logB * logC + 2 * n * (2 * pi_alpha - 1)**2 * np.log(3/2)
        term2 = np.exp(0.5 * logA - 0.5 * np.sqrt(term2_inside_sqrt))
        delta = min(max(term1, term2),1)


        gamma = delta
        tau = np.sqrt((np.log(2*n/gamma))/m)
        inv_tau = 1/tau

        # Round to nearest power of 2
        nearest_pow2 = 2**int(np.round(np.log2(inv_tau)))

        # Adjusted tau
        tau_adj = 1/nearest_pow2
        t0 = time.time()
        est_girgis = meanscalar(user_samples_scaled, tau_adj, alpha, 1.0)
        t1 = time.time()
        girgis_time = t1 - t0
        girgis_mse = float((est_girgis - true_mean_scaled) ** 2)


        row = {
            "n": int(n),
            "m": int(m),
            "alpha": float(alpha),
            "distribution": distribution,
            "true_mean": float(true_mean),
            "seed": int(trial_seed),
            "dame_estimate": float(est_dame),
            "dame_mse": dame_mse,
            "dame_time": float(dame_time),
            "kent_estimate": float(est_kent),
            "kent_mse": kent_mse,
            "kent_time": float(kent_time),
            "girgis_estimate": float(est_girgis),
            "girgis_mse": girgis_mse,
            "girgis_time": float(girgis_time),
            "status": "ok",
        }
    except Exception as e:
        row = {
            "n": int(n) if n is not None else None,
            "m": int(m) if m is not None else None,
            "alpha": float(alpha) if alpha is not None else None,
            "distribution": distribution,
            "true_mean": float(true_mean),
            "seed": int(trial_seed) if trial_seed is not None else None,
            "dame_estimate": None,
            "dame_mse": math.nan,
            "dame_time": math.nan,
            "kent_estimate": None,
            "kent_mse": math.nan,
            "kent_time": math.nan,
            "girgis_estimate": None,
            "girgis_mse": math.nan,
            "girgis_time": math.nan,
            "status": f"error: {repr(e)}",
        }
    return row


def _make_seed(base_seed, param_index, trial_index):
    """
    Generate a reproducible random seed.

    Creates a deterministic 32-bit integer seed derived from the base
    seed, the parameter index, and the trial index. This ensures that
    each (param_value, trial) pair maps to a unique but reproducible
    seed across runs.

    Parameters
    ----------
    base_seed : int
        Global base seed.
    param_index : int
        Index of the parameter value in the grid.
    trial_index : int
        Index of the trial for this parameter value.

    Returns
    -------
    seed : int
        Deterministic 32-bit integer.
    """
    s = f"{base_seed}_{param_index}_{trial_index}"
    h = hashlib.md5(s.encode("utf8")).hexdigest()
    return int(h[:8], 16)


def run_param(
    param_name,
    param_values,
    fixed_n,
    fixed_m,
    fixed_alpha,
    distribution,
    true_mean,
    trials_per_setting=50,
    base_seed=42,
    out_csv_path="results_param.csv",
    n_jobs=8,
):

    """
    Run multiple experimental trials varying one parameter.

    For each value in `param_values`, this function runs
    `trials_per_setting` independent trials in parallel (using a process pool),
    evaluates both DAME-BS and Kent estimators, and writes results to CSV.
    Summary statistics (median, quantile error rates and computation time) are also returned.

    Parameters
    ----------
    param_name : {"alpha", "n", "m"}
        Which parameter to vary across the experiment among privacy parameter (alpha), number of users (n) or number of samples per users (m)
    param_values : list
        List of values for the chosen parameter.
    fixed_n : int
        Fixed number of users (used when param_name is not "n").
    fixed_m : int
        Fixed number of samples per user (used when param_name is not "m").
    fixed_alpha : float
        Fixed privacy/confidence parameter (used when param_name is not "alpha").
    distribution : str
        Distribution type for synthetic data ("uniform","normal","binomial","standard_t")
    true_mean : float
        Ground-truth mean of the generating distribution.
    trials_per_setting : int, optional (default=50)
        Number of independent trials per parameter value.
    base_seed : int, optional (default=42)
        Base seed for reproducibility. It is combined with indices to generate per-trial seeds.
    out_csv_path : str, optional (default="results_param.csv")
        Path to write per-trial results in CSV format.
    n_jobs : int, optional (default=8)
        Number of parallel worker processes.

    Returns
    -------
    results : dict
        A dictionary with keys:
        - "param_values" : list of parameter values
        - "median_dame", "lower10_dame", "upper90_dame"
        - "median_kent", "lower10_kent", "upper90_kent"
        - "median_girgis","lower10_girgis","upper90_girgis"
        - "df" : pandas DataFrame containing all per-trial rows
    """
    if param_name not in {"alpha", "n", "m"}:
        raise ValueError("param_name must be one of 'alpha','n','m'")

    # CSV column order
    csv_columns = [
        "param_name",
        "param_value",
        "n",
        "m",
        "alpha",
        "distribution",
        "true_mean",
        "seed",
        "dame_estimate",
        "dame_mse",
        "dame_time",
        "kent_estimate",
        "kent_mse",
        "kent_time",
        "girgis_estimate",
        "girgis_mse",
        "girgis_time",
        "status",
    ]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # create CSV and write header
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # build tasks: list of tuples (param_index, param_value, trial_index, n, m, alpha, seed)
    tasks = []
    for p_idx, p_val in enumerate(param_values):
        for t_idx in range(trials_per_setting):
            n = fixed_n
            m = fixed_m
            alpha = fixed_alpha
            if param_name == "alpha":
                alpha = float(p_val)
            elif param_name == "n":
                n = int(p_val)
            elif param_name == "m":
                m = int(p_val)
            seed = _make_seed(base_seed, p_idx, t_idx)
            tasks.append((p_idx, p_val, t_idx, n, m, alpha, distribution, true_mean, seed))

    results = []
    total = len(tasks)
    # use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        future_to_task = {
            exe.submit(single_trial, n, m, alpha, distribution, true_mean, seed): (p_idx, p_val)
            for (p_idx, p_val, t_idx, n, m, alpha, distribution, true_mean, seed) in tasks
        }

        for fut in tqdm(as_completed(future_to_task), total=total, desc="running", unit="trial"):
            task_info = future_to_task[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "n": None,
                    "m": None,
                    "alpha": None,
                    "distribution": distribution,
                    "true_mean": true_mean,
                    "seed": None,
                    "dame_estimate": None,
                    "dame_mse": math.nan,
                    "dame_time": math.nan,
                    "kent_estimate": None,
                    "kent_mse": math.nan,
                    "kent_time": math.nan,
                    "girgis_estimate": None,
                    "girgis_mse": math.nan,
                    "girgis_time": math.nan,
                    "status": f"fatal_error: {repr(e)}",
                }

            # determine param_value for this row (from n,m,alpha)
            param_value = row["alpha"] if param_name == "alpha" else (row["n"] if param_name == "n" else row["m"])

            out_row = {
                "param_name": param_name,
                "param_value": param_value,
                "n": row.get("n"),
                "m": row.get("m"),
                "alpha": row.get("alpha"),
                "distribution": row.get("distribution"),
                "true_mean": row.get("true_mean"),
                "seed": row.get("seed"),
                "dame_estimate": row.get("dame_estimate"),
                "dame_mse": row.get("dame_mse"),
                "dame_time": row.get("dame_time"),
                "kent_estimate": row.get("kent_estimate"),
                "kent_mse": row.get("kent_mse"),
                "kent_time": row.get("kent_time"),
                "girgis_estimate": row.get("girgis_estimate"),
                "girgis_mse":row.get("girgis_mse"),
                "girgis_time": row.get("girgis_time"),
                "status": row.get("status", "ok"),
            }

            
            # append to CSV incrementally
            with open(out_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerow(out_row)

            results.append(out_row)

    # aggregate into dataframe
    df = pd.DataFrame(results)
    grouped = df.groupby("param_value")

    median_dame = grouped["dame_mse"].median().reindex(param_values).tolist()
    lower10_dame = grouped["dame_mse"].quantile(0.10).reindex(param_values).tolist()
    upper90_dame = grouped["dame_mse"].quantile(0.90).reindex(param_values).tolist()

    median_kent = grouped["kent_mse"].median().reindex(param_values).tolist()
    lower10_kent = grouped["kent_mse"].quantile(0.10).reindex(param_values).tolist()
    upper90_kent = grouped["kent_mse"].quantile(0.90).reindex(param_values).tolist()

    median_girgis = grouped["girgis_mse"].median().reindex(param_values).tolist()
    lower10_girgis = grouped["girgis_mse"].quantile(0.10).reindex(param_values).tolist()
    upper90_girgis = grouped["girgis_mse"].quantile(0.90).reindex(param_values).tolist()


    return {
        "param_values": list(param_values),
        "median_dame": median_dame,
        "lower10_dame": lower10_dame,
        "upper90_dame": upper90_dame,
        "median_kent": median_kent,
        "lower10_kent": lower10_kent,
        "upper90_kent": upper90_kent,
        "median_girgis":median_girgis,
        "lower10_girgis":lower10_girgis,
        "upper90_girgis":upper90_girgis,
        "df": df,
    }




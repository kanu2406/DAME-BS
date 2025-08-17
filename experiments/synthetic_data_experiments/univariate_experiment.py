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
from dame_bs.utils import plot_errorbars
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator



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
    np.random.seed(seed)

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
        - "dame_estimate", "dame_mse", "dame_time"
        - "kent_estimate", "kent_mse", "kent_time"
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

    return {
        "param_values": list(param_values),
        "median_dame": median_dame,
        "lower10_dame": lower10_dame,
        "upper90_dame": upper90_dame,
        "median_kent": median_kent,
        "lower10_kent": lower10_kent,
        "upper90_kent": upper90_kent,
        "df": df,
    }

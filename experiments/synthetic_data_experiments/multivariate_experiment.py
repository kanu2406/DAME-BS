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
from concurrent.futures import TimeoutError as FutTimeout
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from dame_bs.multivariate_dame_bs import multivariate_dame_bs_l_inf
from kent.multivariate_kent import kent_multivariate_estimator
from girgis.multivariate import *
import warnings
warnings.filterwarnings("ignore")


def generate_multivariate_scaled_data(distribution, n, m,d, true_mean,seed=42):
    """
    Generates and scales multivariate samples into [-1,1]^d.

    Parameters
    ----------
    distribution : str
        Supported Distributions {"normal","uniform","standard_t","binomial"}.
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

    r= np.random.default_rng(seed)
    true_mean = np.asarray(true_mean, dtype=float)
    

    # Generating raw samples of shape (n, m, d)
    if distribution == "normal":
        user_samples = r.normal(loc=true_mean, scale=1.0, size=(n, m, d))
    elif distribution == "uniform":
        user_samples = r.uniform(low=true_mean - 1, high=true_mean + 1, size=(n, m, d))
    elif distribution == "standard_t":
        user_samples = r.standard_t(df=2, size=(n, m, d)) + true_mean
    elif distribution == "binomial":
        trials=50
        p = true_mean # for true_mean in (0,1)
        user_samples = r.binomial(trials,p,size=(n, m,d)).astype(float)
        actual_mean=trials*p
        user_samples = user_samples / float(trials)   # now nominally in [0,1]
        true_mean = actual_mean / float(trials)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Compute per‐coordinate min and max across all users and samples
    # vmin = user_samples.min(axis=(0, 1))  # shape (d,)
    # vmax = user_samples.max(axis=(0, 1))  # shape (d,)
    

    
    # optional: if you want deterministic bounds for binomial proportions, you could override:
    if distribution == "binomial":
        vmin = 0
        vmax = 1
    else:
        vmin = np.min(user_samples)
        vmax = np.max(user_samples)
        denom = vmax-vmin
        # broadcast subtraction/division across axes 0 and 1
        user_samples = (user_samples - vmin) / denom
        true_mean = (true_mean - vmin) / denom

    
    # data_clipped = np.clip(user_samples, 0, 1)
    # user_samples=data_clipped
    vmin,vmax=0,1
    # vmin = user_samples.min()
    # vmax = user_samples.max()
    # jitter_eps = 1e-6
    # if jitter_eps and jitter_eps > 0:
    # noise = np.random.uniform(low=-jitter_eps, high=jitter_eps, size=user_samples.shape)
    # user_samples = user_samples + noise
    # data_clipped = np.clip(user_samples, vmin, vmax)
    # user_samples=data_clipped

    eps=1e-5
    
    rng = vmax - vmin
    # user_samples_scaled = np.zeros_like(user_samples)
    # true_mean_scaled = np.zeros_like(true_mean)

    
    # for k in range(d):
    if rng == 0:
        # if all values of that coordinate are equal
        print(f"Generated values of the coordinate {d} are equal so mapping all of them to zero.")
        # user_samples_scaled[:, :, k] = 0.0
        # true_mean_scaled[k]      = 0.0
        user_samples_scaled = np.zeros_like(user_samples)
        true_mean_scaled = np.zeros_like(true_mean)
    else:
        # safe denominator
        # safe_rng = rng[k] if rng[k] > eps else eps
        safe_rng = rng if rng > eps else eps
        # scaling in [-1,1]^d
        # user_samples_scaled[:, :, k] = (2 * (user_samples[:, :, k] - vmin[k]) / safe_rng) - 1
        # true_mean_scaled[k]  = (2 * (true_mean[k]    - vmin[k]) / safe_rng )- 1
        user_samples_scaled = (2 * (user_samples - vmin) / safe_rng) - 1
        true_mean_scaled  = (2 * (true_mean    - vmin) / safe_rng )- 1



    return user_samples_scaled, true_mean_scaled





def single_trial(n, m, alpha,d, distribution, true_mean, trial_seed):
    """
    Run one complete multivariate experimental trial.

    This function generates synthetic multivariate data, runs both the
    DAME-BS and Kent estimators, and collects their performance metrics
    (mean squared error and runtime). Results are returned in a dict that
    can be written directly to CSV.

    Parameters
    ----------
    n : int
        Number of users.
    m : int
        Number of samples per user.
    alpha : float
        Privacy parameter for the estimators.
    d : int
        Dimension of the multivariate distribution.
    distribution : str
        Distribution type used for synthetic data generation
        (supported values : "normal","uniform","binomial","standard_t").
    true_mean : array-like of shape (d,)
        Ground-truth mean vector of the generating distribution
        (before scaling).
    trial_seed : int
        Random seed controlling reproducibility of this trial.

    Returns
    -------
    row : dict
        Dictionary with per-trial results containing keys:
        - "n", "m", "alpha", "d", "distribution", "true_mean", "seed"
        - "dame_estimate", "dame_mse", "dame_time"
        - "kent_estimate", "kent_mse", "kent_time"
        - "girgis_estimate", "girgis_mse", "girgis_time"
        - "status" : "ok" or error string
    """
    try:
        # Generate data (uses seed)
        X_scaled, true_mean_scaled = generate_multivariate_scaled_data(
            distribution=distribution, n=n, m=m, true_mean=true_mean,d=d, seed=int(trial_seed)
        )

        # run DAME-BS 
        t0 = time.time()
        est_dame = multivariate_dame_bs_l_inf(X_scaled, alpha)
        t1 = time.time()
        dame_time = t1 - t0
        dame_mse = np.linalg.norm(est_dame - true_mean_scaled)**2

        # run Kent
        t0 = time.time()
        est_kent = kent_multivariate_estimator(X_scaled, alpha)
        t1 = time.time()
        kent_time = t1 - t0
        kent_mse = np.linalg.norm(est_kent - true_mean_scaled)**2

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
        tau_adj = 1/nearest_pow2

        t0 = time.time()
        est_girgis = mean_vector(X_scaled, tau, alpha,1.0, gamma)
        t1 = time.time()
        girgis_time = t1 - t0
        girgis_mse = np.linalg.norm(est_girgis - true_mean_scaled)**2


        row = {
            "n": int(n),
            "m": int(m),
            "alpha": float(alpha),
            "d":int(d),
            "distribution": distribution,
            "true_mean": true_mean,
            "seed": int(trial_seed),
            "dame_estimate": est_dame,
            "dame_mse": dame_mse,
            "dame_time": float(dame_time),
            "kent_estimate": est_kent,
            "kent_mse": kent_mse,
            "kent_time": float(kent_time),
            "girgis_estimate": est_girgis,
            "girgis_mse": girgis_mse,
            "girgis_time": float(girgis_time),
            "status": "ok",
        }
    except Exception as e:
        row = {
            "n": int(n) if n is not None else None,
            "m": int(m) if m is not None else None,
            "alpha": float(alpha) if alpha is not None else None,
             "d":int(d) if d is not None else None,
            "distribution": distribution,
            "true_mean": true_mean,
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



# deterministic seed mixing helper
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



def run_param_multivariate(
    param_name,
    param_values,
    fixed_n,
    fixed_m,
    fixed_alpha,
    fixed_d,
    distribution,
    true_mean,
    trials_per_setting=50,
    base_seed=42,
    out_csv_path="results_param.csv",
    n_jobs=8,
):
    """
    Parallel experiment runner for multivariate trials.

    Varies one parameter (`alpha`, `n`, `m`, or `d`) across a grid of values,
    runs multiple independent trials in parallel, evaluates both estimators
    (DAME-BS and Kent), writes results incrementally to CSV, and summarizes
    error statistics.

    Parameters
    ----------
    param_name : {"alpha", "n", "m", "d"}
        Which parameter to vary across the experiment.
    param_values : list
        Sequence of values for the chosen parameter.
    fixed_n : int
        Fixed number of users (used when param_name is not "n").
    fixed_m : int
        Fixed number of samples per user (used when param_name is not "m").
    fixed_alpha : float
        Fixed privacy/confidence parameter (used when param_name is not "alpha").
    fixed_d : int
        Fixed data dimensionality (used when param_name is not "d").
    distribution : str
        Distribution type used to generate synthetic multivariate data.
    true_mean : array-like
        Ground-truth mean vector of the generating distribution.
        If `param_name == "d"`, this is reset automatically to `[0.1] * d`.
    trials_per_setting : int, optional (default=50)
        Number of independent trials per parameter value.
    base_seed : int, optional (default=42)
        Base seed for reproducibility; combined with indices to generate per-trial seeds.
    out_csv_path : str, optional (default="results_param.csv")
        Path to write per-trial results in CSV format (overwritten if exists).
    n_jobs : int, optional (default=8)
        Number of worker processes to use.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "param_values" : list of parameter values (in original order)
        - "median_dame", "lower10_dame", "upper90_dame"
        - "median_kent", "lower10_kent", "upper90_kent"
        - "df" : pandas DataFrame with all per-trial rows
    """
    if param_name not in {"alpha", "n", "m","d"}:
        raise ValueError("param_name must be one of 'alpha','n','m','d'")

    # CSV column order
    csv_columns = [
        "param_name",
        "param_value",
        "n",
        "m",
        "alpha",
        "d",
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

    # create CSV and write header
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
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
            d = fixed_d
            if param_name == "alpha":
                alpha = float(p_val)
            elif param_name == "n":
                n = int(p_val)
            elif param_name == "m":
                m = int(p_val)
            elif param_name == "d":
                d = int(p_val)
                true_mean = [0.1]*d
                
            seed = _make_seed(base_seed, p_idx, t_idx)
            tasks.append((p_idx, p_val, t_idx, n, m, alpha,d, distribution, true_mean, seed))

    results = []
    total = len(tasks)
    # use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        

        future_to_task = {
            exe.submit(single_trial, n, m, alpha,d, distribution, true_mean, seed): (p_idx, p_val)
            for (p_idx, p_val, t_idx, n, m, alpha,d, distribution, true_mean, seed) in tasks
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
                    "d":None,
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

            # determine param_value 
            if param_name == "alpha":
                param_value = row.get("alpha")
            elif param_name == "n":
                param_value = row.get("n")
            elif param_name == "m":
                param_value = row.get("m")
            else:
                param_value = row.get("d")

            out_row = {
                "param_name": param_name,
                "param_value": param_value,
                "n": row.get("n"),
                "m": row.get("m"),
                "alpha": row.get("alpha"),
                "d":row.get("d"),
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
        "median_girgis": median_girgis,
        "lower10_girgis": lower10_girgis,
        "upper90_girgis": upper90_girgis,
        "df": df,
    }




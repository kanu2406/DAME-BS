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
from Backtrack.dame_bs_backtrack import dame_with_binary_search_backtrack




def generate_univariate_scaled_data(distribution,n,m, true_mean,seed=42):

    
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
        p = true_mean
        trials = 50
        user_samples = r.binomial(trials, p, size=(n, m)).astype(float)
        user_samples = user_samples / float(trials)   # now nominally in [0,1]
        actual_mean = trials*p
        true_mean = actual_mean / float(trials)


    
    
    if distribution == "binomial":
        vmin,vmax = 0,1
    else:
        vmin = np.min(user_samples)
        vmax = np.max(user_samples)
        user_samples = (user_samples-vmin) / (vmax-vmin)   # now nominally in [0,1]
        true_mean = (true_mean-vmin) /(vmax-vmin)

    data_clipped = np.clip(user_samples, vmin, vmax)
    jitter_eps = 1e-6
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










def plot_errorbars(x_values, median_errors_dame_bs,lower_errors_dame_bs, upper_errors_dame_bs,
                   median_errors_dame_backtrack,lower_errors_dame_backtrack,upper_errors_dame_backtrack, xlabel, 
                   ylabel, title,log_scale=True,plot_ub=False,upper_bounds=None,save_path=None,
                   log_log_scale = False,y_lim=True):
   

    if upper_bounds is None:
        upper_bounds=[]
    plt.figure(figsize=(8, 5))
    plt.fill_between(x_values, lower_errors_dame_bs, upper_errors_dame_bs, alpha=0.3)
    plt.plot(x_values, median_errors_dame_bs,label="DAME-BS")
    
    plt.fill_between(x_values, lower_errors_dame_backtrack, upper_errors_dame_backtrack, alpha=0.3)
    plt.plot(x_values, median_errors_dame_backtrack,label="DAME-BS Backtrack")
    
    all_lowers = np.minimum( lower_errors_dame_bs, lower_errors_dame_backtrack)
    all_uppers = np.maximum( upper_errors_dame_bs, upper_errors_dame_backtrack)

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

       
        # run DAME-BS with Backtrack
        t0 = time.time()
        est_dame_backtrack = dame_with_binary_search_backtrack(n, alpha, m, user_samples_scaled)
        t1 = time.time()
        dame_backtrack_time = t1 - t0
        dame_backtrack_mse = float((est_dame_backtrack - true_mean_scaled) ** 2)


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
            "dame_backtrack_estimate": float(est_dame_backtrack),
            "dame_backtrack_mse": dame_backtrack_mse,
            "dame_backtrack_time": float(dame_backtrack_time),
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
            "dame_backtrack_estimate": None,
            "dame_backtrack_mse": math.nan,
            "dame_backtrack_time": math.nan,
            "status": f"error: {repr(e)}",
        }
    return row


def _make_seed(base_seed, param_index, trial_index):
    
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
        "dame_backtrack_estimate",
        "dame_backtrack_mse",
        "dame_backtrack_time",
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
                    "dame_backtrack_estimate": None,
                    "dame_backtrack_mse": math.nan,
                    "dame_backtrack_time": math.nan,
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
                "dame_backtrack_estimate": row.get("dame_backtrack_estimate"),
                "dame_backtrack_mse":row.get("dame_backtrack_mse"),
                "dame_backtrack_time": row.get("dame_backtrack_time"),
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

    median_dame_backtrack = grouped["dame_backtrack_mse"].median().reindex(param_values).tolist()
    lower10_dame_backtrack = grouped["dame_backtrack_mse"].quantile(0.10).reindex(param_values).tolist()
    upper90_dame_backtrack = grouped["dame_backtrack_mse"].quantile(0.90).reindex(param_values).tolist()


    return {
        "param_values": list(param_values),
        "median_dame": median_dame,
        "lower10_dame": lower10_dame,
        "upper90_dame": upper90_dame,
        "median_dame_backtrack":median_dame_backtrack,
        "lower10_dame_backtrack":lower10_dame_backtrack,
        "upper90_dame_backtrack":upper90_dame_backtrack,
        "df": df,
    }




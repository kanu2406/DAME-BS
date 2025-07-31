import numpy as np
import pandas as pd
import sys,os,random,time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from kent.kent import kent_mean_estimator
from dame_bs.dame_bs import dame_with_binary_search
from experiments.real_data_experiments.globem.preprocess import load_and_select_sleep,truncating_data,scaling

def main():
    """
    Main experiment function for evaluating Kent's algorithm and DAME-BS 
    on GLOBEM sleep duration data.

    Steps:
    -------
        - Loads and concatenates multiple CSV files containing Fitbit-derived sleep duration.
        - Handles missing data and prepares a uniform sample length across users.
        - Scales the sleep duration values to the range [-1, 1].
        - Runs both algorithms 500 times to estimate the mean and records time and accuracy.
        - Reports mean squared error (MSE) in both scaled and original ranges, along with timing stats.
    """

    # Loading Data
    feature_dir = "experiments/Datasets/GLOBEM_Sleep"
    files = [os.path.join(feature_dir, 'sleep_sample_1.csv'),os.path.join(feature_dir, 'sleep_sample_2.csv'),
         os.path.join(feature_dir, 'sleep_sample_3.csv'),os.path.join(feature_dir, 'sleep_sample_4.csv')]

    dfs = []
    for path in files:
        dfs.append(load_and_select_sleep(path))
    # concatinating data from different files into one dataframe
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.sort_values(['pid','date']).reset_index(drop=True)
    print("Loaded Data.")
    print(df_all.shape)
    print(df_all.head())

    # count NaNs per column
    nan_per_col = df_all.isna().sum()
    print("Number of missing value per column : ",nan_per_col)

    # count NaNs per row
    n_nan_rows = df_all.isna().any(axis=1).sum()
    print("Rows with ≥1 NaN:", n_nan_rows)

    # Grouping by pid
    pid_and_sleep = {}
    for pid, group in df_all.groupby('pid'):
        arr = group['sleep_duration'].values
        flat = arr.flatten().tolist()
        pid_and_sleep[pid] = flat

    # Removing NaNs from each user's list
    pid_and_sleep_new = {
        pid: [v for v in vals if not pd.isna(v)]
        for pid, vals in pid_and_sleep.items()
    }

    # Truncating samples so that each user has same number of samples
    pid_and_sleep_final,desired_length = truncating_data(pid_and_sleep_new)
    print("Truncation Done.")

    # Scaling data in the range [-1,1]
    user_samples_scaled, min_val,max_val,true_mean,true_mean_scaled = scaling(pid_and_sleep_final)
    print("Scaling Done.")

    print("Total number of users", len(user_samples_scaled))
    print("Number of samples per user : ",desired_length)
    print(f"True mean : {true_mean:.3f}" )
    print(f"True mean scaled : {true_mean_scaled:.3f}" )

    random.shuffle(user_samples_scaled)
    
    print("Running both algorithms 500 times")
    n = len(user_samples_scaled)
    m = desired_length
    alpha = 0.6
    X=np.array(user_samples_scaled)

    num_exp = 500
    errors_kent = []
    errors_dame_bs = []
    true_err_kent = []
    true_err_dame_bs = []
    time_kent = []
    time_dame_bs = []
    theta_hats_kent = []
    theta_hats_dame_bs = []
    ests_kent = []
    ests_dame_bs = []

    for _ in range(num_exp):
        
        start_time_kent = time.time()
        theta_hat_kent   = kent_mean_estimator(X, alpha=0.6, K=1.0)
        time_kent.append(time.time() - start_time_kent)
        est_kent = 0.5 * (theta_hat_kent + 1) * (max_val - min_val) + min_val #estimated mean in the orignal range
        theta_hats_kent.append(theta_hat_kent)
        ests_kent.append(est_kent)

        start_time_dame_bs = time.time()
        theta_hat_dame_bs   = dame_with_binary_search(n, alpha, m, user_samples_scaled)
        time_dame_bs.append(time.time() - start_time_dame_bs)
        est_dame_bs = 0.5 * (theta_hat_dame_bs + 1) * (max_val - min_val) + min_val #estimated mean in the orignal range
        theta_hats_dame_bs.append(theta_hat_dame_bs)
        ests_dame_bs.append(est_dame_bs)

        errors_kent.append( (theta_hat_kent - true_mean_scaled)**2 )
        errors_dame_bs.append( (theta_hat_dame_bs - true_mean_scaled)**2 )

        true_err_kent.append( (est_kent - true_mean)**2 )
        true_err_dame_bs.append( (est_dame_bs - true_mean)**2 )


     
    print("--------------------------STATS FOR KENT'S Algo --------------------------------")
    print(f"Time taken: {np.mean(time_kent):.5f}s")
    print(f"Estimated Mean in the range [-1,1]= {np.mean(theta_hats_kent):.4f}, true mean scaled = {true_mean_scaled:.4f}")
    print(f"Estimated mean = {np.mean(ests_kent):.6f}, true mean = {true_mean:.6f}")
    print(f"MSE between values in range [-1,1] (scaled) = {np.mean(errors_kent):.6f}")
    print(f"MSE (in original range) = {np.mean(true_err_kent):.6f}")
    print("--------------------------------------------------------------------------------")

    
    print("----------------------------STATS FOR DAME-BS ----------------------------------")
    print(f"Time taken: {np.mean(time_dame_bs):.5f}s")
    print(f"Estimated Mean in the range [-1,1]= {np.mean(theta_hats_dame_bs):.4f}, true mean scaled = {true_mean_scaled:.4f}")
    print(f"Estimated mean = {np.mean(ests_dame_bs):.6f}, true mean = {true_mean:.6f}")
    print(f"MSE between values in range [-1,1] (scaled) = {np.mean(errors_dame_bs):.6f}")
    print(f"MSE (in original range) = {np.mean(true_err_dame_bs):.6f}")
    print("--------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
    



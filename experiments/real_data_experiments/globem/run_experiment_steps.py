import numpy as np
import pandas as pd
import sys,os,random,time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from kent.kent import kent_mean_estimator
from dame_bs.dame_bs import dame_with_binary_search
from experiments.real_data_experiments.globem.preprocess import load_and_select_steps,truncating_data,scaling

def main():

    """
    Executes the mean estimation experiment on segmented step count data.

    Steps:
    -------
        - Load 4 CSV files containing Fitbit-derived step counts over different time segments of the day.
        - Preprocess the data: merge, remove NaNs, flatten per-user data, and truncate to uniform length.
        - Scale data to the [-1, 1] range and compute true means.
        - Shuffle users and run both estimation algorithms (Kent's and DAME-BS) 500 times.
        - Record and print:
            - Mean estimated values
            - Mean Squared Error (MSE) on scaled and unscaled data
            - Average runtime per algorithm
    """

    # Loading Data
    feature_dir = "experiments/Datasets/GLOBEM_Steps"
    files = [os.path.join(feature_dir, 'steps_sample_1.csv'),os.path.join(feature_dir, 'steps_sample_2.csv'),
            os.path.join(feature_dir, 'steps_sample_3.csv'),os.path.join(feature_dir, 'steps_sample_4.csv')]
    dfs = []
    
    for path in files:
        dfs.append(load_and_select_steps(path))
    # concatinating data from different files into one dataframe
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.sort_values(['pid','date']).reset_index(drop=True)
    print("Loaded Data.")
    print(df_all.shape)
    print(df_all.head())

    # count NaNs per column
    nan_per_col = df_all.isna().sum()
    print(nan_per_col)

    # count NaNs per row
    n_nan_rows = df_all.isna().any(axis=1).sum()
    print("Rows with ≥1 NaN:", n_nan_rows)


    # each user (pid) has a step count in four segments 
    seg_cols = ['steps_morning','steps_afternoon','steps_evening','steps_night']
    # Group by pid and collect all segment values into a single list per pid
    pid_and_steps = {}
    for pid, group in df_all.groupby('pid'):
        # extract the numpy array of shape (num_days, 4)
        arr = group[seg_cols].values
        # flatten to 1D and convert to Python list
        flat = arr.flatten().tolist()
        pid_and_steps[pid] = flat

    # Removing Nan values from data
    pid_and_steps_new = {}
    for pid, vals in pid_and_steps.items():
        pid_and_steps_new[pid] = [v for v in vals if not pd.isna(v)]

    # Truncating samples so that each user has same number of samples
    pid_and_steps_final,desired_length = truncating_data(pid_and_steps_new)
    print("Truncation Done.")

    # Scaling data in the range [-1,1]
    user_samples_scaled, min_val,max_val,true_mean,true_mean_scaled = scaling(pid_and_steps_final)
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
    



import pandas as pd
import numpy as np
import time,random
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
from experiments.real_data_experiments.mimic.preprocess import truncate_and_shuffle,scaling_data
from dame_bs.dame_bs import dame_with_binary_search
from kent.kent import kent_mean_estimator
from collections import defaultdict

np.random.seed(42)
def main():
    """
    This script runs a 500-trial comparison of Kent's mean estimator and DAME-BS
    on real heart rate data from the MIMIC-III dataset.

    Steps:
        - Loads and filters MIMIC-III CHARTEVENTS for heart rate data.
        - Scales all values to [-1, 1] and truncates samples so each user have same number of samples.
        - Runs both algorithms across 500 trials.
        - Reports runtime, mean estimates, and MSE (in both scaled and original ranges).

    """

    # Load patients data 
    df = pd.read_csv("experiments/Datasets/CHARTEVENTS.csv", low_memory=False)

    # Filter for heart rate (i.e. itemid = 211) and non-missing values
    df_hr = df[(df['itemid'] == 211) & (df['valuenum'].notna())]
    # Keep only relevant columns
    df_hr = df_hr[['subject_id','itemid', 'charttime', 'valuenum']]
    print(df_hr.head())

    # converting to dictionary
    user_samples = defaultdict(list)
    for row in df_hr.itertuples():
        sid = row.subject_id
        if sid not in user_samples:
            user_samples[sid] = []
        user_samples[sid].append(row.valuenum)
    
    # keeping only SIDs with at least 7 samples
    user_samples = {sid: vals for sid, vals in user_samples.items() if len(vals) >= 7}

    print("Data Loaded.")

    # Scaling everything to range [-1,1]
    user_samples_scaled,desired_length,true_mean,true_mean_scaled,vmin,vmax=scaling_data(user_samples)
    print("Scaling Done.")

    # Truncating and shuffling 
    users_and_heart_rate_final = truncate_and_shuffle(user_samples_scaled,desired_length)
    print("Truncation Done.")
    print("Total number of users", len(user_samples))
    print("Number of samples per user : ",desired_length)
    print(f"True mean : {true_mean:.3f}" )
    print(f"True mean scaled : {true_mean_scaled:.3f}" )
    
    # Flattening the data to list 
    heart_rates=[vals for vals in users_and_heart_rate_final.values()]
    random.shuffle(heart_rates)
    
    print("Running both algorithms 500 times")
    n = len(heart_rates)
    m = desired_length
    alpha = 0.6
    X=np.array(heart_rates)

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
        est_kent = 0.5 * (theta_hat_kent + 1) * (vmax - vmin) + vmin #estimated mean in the orignal range
        theta_hats_kent.append(theta_hat_kent)
        ests_kent.append(est_kent)

        start_time_dame_bs = time.time()
        theta_hat_dame_bs   = dame_with_binary_search(n, alpha, m, X)
        time_dame_bs.append(time.time() - start_time_dame_bs)
        est_dame_bs = 0.5 * (theta_hat_dame_bs + 1) * (vmax - vmin) + vmin #estimated mean in the orignal range
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


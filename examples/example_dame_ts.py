# Example to run dame_ts and get the mean estimation


from dame_ts.dame_ts import dame_with_ternary_search
import numpy as np
import math 

if __name__ == "__main__":
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 20000
    m = 20
    true_mean = 0.3
    user_samples= [np.random.normal(loc=true_mean, scale=0.6, size=m) for _ in range(n)]
    
    bar_theta = dame_with_ternary_search(n, alpha, m, user_samples)
    print(f"Final mean Estimate: {bar_theta:.3f}")
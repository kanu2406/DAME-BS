# Example to run ternary search and get the estimated interval for mean estimation

from dame_ts.ternary_search import attempting_insertion_using_ternary_search
import numpy as np
import math 

if __name__ == "__main__":
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples= [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    L, R = attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)
    print(f"Ternary Search Interval: [{L:.3f}, {R:.3f}]")
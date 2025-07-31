import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.univariate_experiment import experiment_risk_vs_param_for_dist_univariate
from experiments.multivariate_experiment import experiment_risk_vs_param_for_dist_multivariate


np.random.seed(42)

def main():
    """
    Main execution function that runs a series of experiments across:
    - Univariate distributions (Normal, Uniform, Binomial)
    - Multivariate distributions with varying dimensions

    Each distribution is evaluated under variation for different value of following parameters:
    - `alpha` (privacy parameter)
    - `n` (number of users)
    - `m` (number of samples per user)
    - `d` (dimensionality)

    The true mean used for comparison is fixed (scalar for univariate, vector for multivariate).
    Each experiment is repeated for a fixed number of trials = 200.
    """
    distributions = ["normal", "uniform","binomial"]
    n_values = list(range(40, 10000  , 400))
    m_values = list(range(1, 500, 60))
    alpha_values = np.linspace(0.1, 3.0,15 )
    d_values = list(range(2,200,20))
    true_mean = 0.3
    
    print("Running univariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="alpha",param_values=alpha_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=200)
    
    print("Running univariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="n",param_values=n_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=200)
    
    print("Running univariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="m",param_values=m_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=200)
        
    
    print("Running multivariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="alpha",param_values=alpha_values,
                                      n=8000,m=20,d=4,alpha=0.6,true_mean=[0.1]*4,trials=200)
        
    print("Running multivariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="n",param_values=n_values,
                                      n=8000,m=20,d=4,alpha=0.6,true_mean=[0.1]*4,trials=200)
    
    print("Running multivariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="m",param_values=m_values,
                                      n=8000,m=20,d=4,alpha=0.6,true_mean=[0.1]*4,trials=200)
    
    print("Running univariate experiments for different distributions for various values of d.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="d",param_values=d_values,
                                      n=8000,m=20,d=4,alpha=0.6,true_mean=[0.1]*4,trials=200)
    

if __name__ == "__main__":
    main()

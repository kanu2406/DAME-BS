# run_experiments.py
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.univariate_experiment import experiment_risk_vs_param_for_dist_univariate
from experiments.multivariate_experiment import experiment_risk_vs_param_for_dist_multivariate
from dame_bs.utils import plot_errorbars  

def main():
    distributions = ["normal", "uniform", "student_t", "binomial"]
    n_values = list(range(1000, 16000  , 2000))
    m_values = list(range(7, 500, 50))
    alpha_values = np.linspace(0.1, 10.0, 15)
    d_values = list(range(2,500,100))
    true_mean = 0.0
    
    print("Running univariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="alpha",param_values=alpha_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=50)
    
    print("Running univariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="n",param_values=n_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=50)
    
    print("Running univariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_univariate(dist,param_to_vary="m",param_values=m_values,
                                      n=8000,m=20,alpha=0.6,true_mean=true_mean,trials=50)
        
    
    print("Running multivariate experiments for different distributions for various alpha values.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="alpha",param_values=alpha_values,
                                      n=8000,m=20,d=10,alpha=0.6,true_mean=[0.1]*10,trials=50)
        
    print("Running multivariate experiments for different distributions for various values of n.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="n",param_values=n_values,
                                      n=8000,m=20,d=10,alpha=0.6,true_mean=[0.1]*10,trials=50)
    
    print("Running multivariate experiments for different distributions for various values of m.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="m",param_values=m_values,
                                      n=8000,m=20,d=10,alpha=0.6,true_mean=[0.1]*10,trials=50)
    
    print("Running univariate experiments for different distributions for various values of d.")
    print("------------------------------------------------------------------------------------")
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_param_for_dist_multivariate(dist,param_to_vary="d",param_values=d_values,
                                      n=8000,m=20,d=10,alpha=0.6,true_mean=[0.1]*10,trials=50)
    

if __name__ == "__main__":
    main()

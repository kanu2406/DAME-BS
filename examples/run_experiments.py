# Example to run DAME-TS algorithm and plot error vs alpha and error vs n for different distributions 


from experiments.risk_vs_alpha import experiment_risk_vs_alpha_for_dist
from experiments.risk_vs_n import experiment_risk_vs_n_for_dist
def run_experiments():
    distributions = ["normal", "uniform", "laplace", "exponential"]
    for dist in distributions:
        print(f"\n Running experiment for distribution: {dist}")
        experiment_risk_vs_alpha_for_dist(dist)
        experiment_risk_vs_n_for_dist(dist)


if __name__ == "__main__":
    run_experiments()


import pytest 
from dame_ts.ternary_search import attempting_insertion_using_ternary_search
from dame_ts.dame_ts import dame_with_ternary_search
from dame_ts.utils import min_n_required
import numpy as np
import math

###########################################################################################
# Tests for output ranges
def test_dame_with_ternary_search_output_range():
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.5
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    estimate = dame_with_ternary_search(n, alpha, m, user_samples)
    assert isinstance(estimate, float)
    assert -1 <= estimate <= 1


def test_ternary_search_output_format():
   
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)

    assert isinstance(interval, list)
    assert len(interval) == 2
    assert isinstance(interval[0], float)
    assert isinstance(interval[1], float)
    assert interval[0] < interval[1]
    assert -1<=interval[0] <=1
    assert -1<=interval[1] <=1


def test_ternary_search_no_nan_inf():
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)

    assert not (math.isnan(interval[0]) or math.isnan(interval[0]))
    assert not (math.isinf(interval[1]) or math.isinf(interval[1]))

def test_ternary_search_interval_bounds():
   
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)

    assert -1<=interval[0] <=1
    assert -1<=interval[1] <=1


##################################################################################################

def test_ternary_search_interval_no_noise():
    
    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    pi_alpha = 1
    two_pi_minus_1 = 2 * pi_alpha - 1

    if abs(two_pi_minus_1) < 1e-6:
        n = np.inf  # Avoid division by near-zero

    ln_32 = np.log(3 / 2)
    term1 = 4 / (two_pi_minus_1 ** 4)
    term2 = np.sqrt(2) + np.sqrt(2 + ln_32 * (two_pi_minus_1 ** 2))
    n = int(np.ceil(term1 * (term2 ** 2))+1000)
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    L, R = attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)
    
    assert L <= true_mean <= R, f"True mean {true_mean} not in estimated interval [{L}, {R}]"
    assert R-L<2, f"estimated interval size has not decreased"

def test_ternary_search_interval_shrinks():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n_small = 5000
    n_large = 15000
    delta_small = 2 * n_small * math.exp(-n_small * (2 * pi_alpha - 1)**2 / 2)
    delta_large = 2 * n_large * math.exp(-n_large * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3


    user_samples_small = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n_small)]
    user_samples_large = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n_large)]

    L1, R1 = attempting_insertion_using_ternary_search(alpha, delta_small, n_small, m, user_samples_small)
    L2, R2 = attempting_insertion_using_ternary_search(alpha, delta_large, n_large, m, user_samples_large)

    width1 = R1 - L1
    width2 = R2 - L2

    assert width2 <= width1, f"Interval did not shrink with more users: {width1:.4f} vs {width2:.4f}"

#########################################################################################################
# Invalid Input Check for ternary search

def test_ternary_search_invalid_n_type():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = -5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)

def test_ternary_search_invalid_alpha_type():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        attempting_insertion_using_ternary_search(-alpha, delta, n, m, user_samples)




def test_ternary_search_mismatch_samples():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n-1)]
    
    with pytest.raises(ValueError, match=f"user_samples must be a list of length {n}"):
        attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)

def test_ternary_search_wrong_sample_length():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m-1) for _ in range(n)]
    with pytest.raises(ValueError, match=f"Each user sample must be an array-like of length {m}"):
        attempting_insertion_using_ternary_search(alpha, delta, n, m, user_samples)


##################################################################################################
# Invalid Input Check for dame_ts

def test_dame_with_ternary_search_invalid_n_type():
    alpha = 0.6
    n = -5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        dame_with_ternary_search(n, alpha, m, user_samples)

def test_dame_with_ternary_search_invalid_alpha_type():
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        dame_with_ternary_search(n, -alpha, m, user_samples)




def test_dame_with_ternary_search_mismatch_samples():
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n-1)]
    
    with pytest.raises(ValueError, match=f"user_samples must be a list of length {n}"):
        dame_with_ternary_search(n, alpha, m, user_samples)

def test_ternary_search_wrong_sample_length():
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m-1) for _ in range(n)]
    with pytest.raises(ValueError, match=f"Each user sample must be an array-like of length {m}"):
        dame_with_ternary_search(n, alpha, m, user_samples)







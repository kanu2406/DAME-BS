'''Unit Tests for dame_bs core functions binary_search() and dame_with_binary_search()'''

import pytest 
from dame_bs.binary_search import attempting_insertion_using_binary_search
from dame_bs.dame_bs import dame_with_binary_search
from dame_bs.utils import theoretical_upper_bound,run_dame_experiment
import numpy as np
import math
import warnings

###########################################################################################
# Tests for output ranges
def test_dame_with_binary_search_output_range():
    """
    Test that dame_with_binary_search returns a float estimate within the valid range [-1, 1].
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.5
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    estimate = dame_with_binary_search(n, alpha, m, user_samples,delta=0.1)
    assert isinstance(estimate, float)
    assert -1 <= estimate <= 1


def test_binary_search_output_format():
    """
    Test that attempting_insertion_using_binary_search returns an interval as a list of two floats.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 0.1

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)

    assert isinstance(interval, list)
    assert len(interval) == 2
    assert isinstance(interval[0], float)
    assert isinstance(interval[1], float)
    assert interval[0] < interval[1]
    


def test_binary_search_no_nan_inf():
    """
    Test that the binary search interval does not contain NaN or infinite values.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    pi_alpha = math.exp(alpha) / (1 + math.exp(alpha))
    delta = 0.1
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)

    assert not (math.isnan(interval[0]) or math.isnan(interval[1]))
    assert not (math.isinf(interval[0]) or math.isinf(interval[1]))

def test_binary_search_interval_bounds():
    """
    Confirm that the interval returned by binary search lies within [-1, 1].
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    delta = 0.1

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    interval = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)

    assert -1<=interval[0] <=1
    assert -1<=interval[1] <=1


##################################################################################################

def test_binary_search_interval_no_noise():
    """
    Test binary search interval estimation with no noise (alpha = infinity).
    Checks that true mean lies within the estimated interval and the interval length has decreased.
    """
    np.random.seed(42)
    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    pi_alpha = 1
    # two_pi_minus_1 = 2 * pi_alpha - 1

    # if abs(two_pi_minus_1) < 1e-6:
    #     n = np.inf  # Avoid division by near-zero

    # ln_32 = np.log(3 / 2)
    # term1 = 4 / (two_pi_minus_1 ** 4)
    # term2 = np.sqrt(2) + np.sqrt(2 + ln_32 * (two_pi_minus_1 ** 2))
    # n = int(np.ceil(term1 * (term2 ** 2))+1000)
    n = 9000
    delta = 0.1
    m = 20
    true_mean = 0.3

    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    L, R = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)
    
    assert L <= true_mean <= R, f"True mean {true_mean} not in estimated interval [{L}, {R}]"
    assert R-L<2, f"estimated interval size has not decreased"

def test_binary_search_interval_shrinks():
    """
    Test that the binary search confidence interval shrinks when number of users increases.
    Compares interval widths for smaller and larger n.
    """
    np.random.seed(42)
    alpha = 0.6
    n_small = 5000
    n_large = 15000
    delta_small=0.1
    delta_large=0.1
    m = 20
    true_mean = 0.3


    user_samples_small = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n_small)]
    user_samples_large = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n_large)]

    L1, R1 = attempting_insertion_using_binary_search(alpha, delta_small, n_small, m, user_samples_small)
    L2, R2 = attempting_insertion_using_binary_search(alpha, delta_large, n_large, m, user_samples_large)

    width1 = R1 - L1
    width2 = R2 - L2

    assert width2 <= width1, f"Interval did not shrink with more users: {width1:.4f} vs {width2:.4f}"



def test_dame_with_binary_search_no_noise():
    """
    Test dame_with_binary_search output when noise is absent (alpha = infinity).
    Checks that estimated mean is within valid range and close to the true mean.
    
    """
    np.random.seed(42)
    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    n=9000
    m = 20
    true_mean = 0.3
    delta=0.1
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    estimated_mean=dame_with_binary_search(n, alpha, m, user_samples,delta)
    assert -1 <= estimated_mean <= 1, f"Estimated mean {estimated_mean} not in interval [-1,1]"
    assert np.abs(estimated_mean-true_mean)**2<= 1e-3, f"Estimated mean and true mean are not close."
    

#########################################################################################################
# Invalid Input Check for binary search

def test_binary_search_invalid_n_type():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for negative n.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, -n, m, user_samples)

def test_binary_search_invalid_alpha_type():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for negative alpha.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        attempting_insertion_using_binary_search(-alpha, delta, n, m, user_samples)




def test_binary_search_mismatch_samples():
    """
    Check that an error is raised if user_samples length does not match n.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n-1)]
    
    with pytest.raises(ValueError, match=f"user_samples must be a list of length {n}"):
        attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)

def test_binary_search_wrong_sample_length():
    """
    Check that an error is raised if individual user samples do not have length m.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m-1) for _ in range(n)]
    with pytest.raises(ValueError, match=f"Each user sample must be an array-like of length {m}"):
        attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)


def test_binary_search_warns_and_corrects_odd_n():
    """
    Check that a warning is issued when n is odd and corrected internally.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n1 = 5001
    n2 = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples1 = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n1)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = attempting_insertion_using_binary_search(alpha, delta, n1, m, user_samples1)  
        assert any("is odd" in str(warning.message) for warning in w)
    
    
def test_binary_search_invalid_m_not_int():
    """
    Check that ValueError is raised when m is not a positive integer.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match=f"m must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, n, "a", user_samples)

def test_binary_search_invalid_delta_not_positive():
    """
    Check that ValueError is raised if delta is not a positive number less than 1.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match=f"delta must be a positive number less than 1"):
        attempting_insertion_using_binary_search(alpha, -delta, n, m, user_samples)

def test_binary_search_invalid_delta_interval():
    """
    Check that ValueError is raised if delta is >= 1.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 2
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match=f"delta must be a positive number less than 1"):
        attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)


def test_binary_search_invalid_m_negative():
    """
    Check that ValueError is raised if m is negative.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match="m must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, n, -m, user_samples)

    

##################################################################################################
# Invalid Input Check for dame_bs

def test_dame_with_binary_search_invalid_n_type():
    """
    Check that dame_with_binary_search raises ValueError for negative n.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        dame_with_binary_search(-n, alpha, m, user_samples,delta=0.1)

def test_dame_with_binary_search_invalid_alpha_type():
    """
    Check that dame_with_binary_search raises ValueError for negative alpha.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        dame_with_binary_search(n, -alpha, m, user_samples,delta=0.1)




def test_dame_with_binary_search_mismatch_samples():
    """
    Check that an error is raised if user_samples length does not match n.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n-1)]
    
    with pytest.raises(ValueError, match=f"user_samples must be a list of length {n}"):
        dame_with_binary_search(n, alpha, m, user_samples,delta=0.1)

def test_dame_with_binary_search_wrong_sample_length():
    """
    Check that an error is raised if individual user samples do not have length m.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m-1) for _ in range(n)]
    with pytest.raises(ValueError, match=f"Each user sample must be an array-like of length {m}"):
        dame_with_binary_search(n, alpha, m, user_samples,delta)



def test_dame_with_binary_search_warns_and_corrects_odd_n():
    """
    Check that a warning is issued when n is odd and corrected internally.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n1 = 5001
    n2 = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples1 = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n1)]
    user_samples2 = user_samples1[:n2]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = dame_with_binary_search(n1, alpha, m, user_samples1,delta)  
        assert any("is odd" in str(warning.message) for warning in w)
    
def test_dame_with_binary_search_invalid_m_not_int():
    """
    Check that ValueError is raised if m (number of samples per user) is not a positive inetger.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match=f"m must be a positive integer"):
        dame_with_binary_search(n, alpha, "a", user_samples,delta)


def test_dame_with_binary_search_invalid_m_negative():
    """
    Check that ValueError is raised if m is negative.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match="m must be a positive integer"):
        dame_with_binary_search(n, alpha, -m, user_samples,delta)

def test_dame_with_binary_search_invalid_delta_negative():
    """
    Check that ValueError is raised if delta is negative.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match="delta must be a positive number less than 1"):
        dame_with_binary_search(n, alpha, m, user_samples,-delta)

def test_dame_with_binary_search_invalid_delta_interval():
    """
    Check that ValueError is raised if delta is not a positive number less than 1.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 2
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match="delta must be a positive number less than 1"):
        dame_with_binary_search(n, alpha, m, user_samples,delta)


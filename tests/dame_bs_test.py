'''Unit Tests for dame_with_binary_search()'''

import pytest 
from dame_bs.binary_search import attempting_insertion_using_binary_search
from dame_bs.dame_bs import dame_with_binary_search
from dame_bs.utils import theoretical_upper_bound
import numpy as np
import math
import warnings


##################################################################################################
# Invalid Input Check for dame_bs

def test_dame_with_binary_search_invalid_n_type1():
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
        dame_with_binary_search(-n, alpha, m, user_samples)


def test_dame_with_binary_search_invalid_n_type2():
    """
    Check that dame_with_binary_search raises ValueError for n non-integer value.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        dame_with_binary_search(n+0.5, alpha, m, user_samples)


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
        dame_with_binary_search(n, -alpha, m, user_samples)




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
        dame_with_binary_search(n, alpha, m, user_samples)





def test_dame_with_binary_search_wrong_sample_length():
    """
    Check that an error is raised if individual user samples do not have length m.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m-1) for _ in range(n)]
    with pytest.raises(ValueError, match=f"Each user sample must be an array-like of length {m}"):
        dame_with_binary_search(n, alpha, m, user_samples)




def test_dame_with_binary_search_warns_and_corrects_odd_n():
    """
    Check that a warning is issued when n is odd and corrected internally.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n1 = 5001
    n2 = 5000
    m = 20
    true_mean = 0.3
    user_samples1 = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n1)]
    user_samples2 = user_samples1[:n2]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = dame_with_binary_search(n1, alpha, m, user_samples1)  
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
        dame_with_binary_search(n, alpha, "a", user_samples)



def test_dame_with_binary_search_invalid_m_negative():
    """
    Check that ValueError is raised if m is negative.
    """
    np.random.seed(42)
    alpha = 0.6
    pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
    n = 5000
    m = 20
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with pytest.raises(ValueError, match="m must be a positive integer"):
        dame_with_binary_search(n, alpha, -m, user_samples)




def test_binary_search_warns_about_smaller_m():
    """
    Check that a warning is issued when n is odd and corrected internally.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 6
    true_mean = 0.3
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = dame_with_binary_search(n, alpha, m, user_samples)  
        assert any("is below the recommended minimum 7" in str(warning.message) for warning in w)
    


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
    estimate = dame_with_binary_search(n, alpha, m, user_samples)
    assert isinstance(estimate, float)
    assert -1 <= estimate <= 1

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
    user_samples = [np.random.normal(loc=true_mean, scale=0.5, size=m) for _ in range(n)]
    estimated_mean=dame_with_binary_search(n, alpha, m, user_samples)
    assert -1 <= estimated_mean <= 1, f"Estimated mean {estimated_mean} not in interval [-1,1]"
    assert np.abs(estimated_mean-true_mean)**2<= 1e-3, f"Estimated mean and true mean are not close."
    





def test_dame_on_constant_data_inf_alpha():
    """
    Test that on constant data and no noise we should be able to get a very good estimate of estimated mean.
    """
    np.random.seed(0)
    n, m = 1000, 20
    c = 0.1
    user_samples = [np.full(m, c) for _ in range(n)]
    est = dame_with_binary_search(n, np.inf, m, user_samples)
    assert abs(est - c) < 1e-3

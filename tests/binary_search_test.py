import pytest 
from dame_bs.binary_search import attempting_insertion_using_binary_search
from experiments.univariate_experiment import generate_univariate_scaled_data
import numpy as np
import math
import warnings

###########################################################################################
# Tests for output ranges


def test_binary_search_output_format():
    """
    Test that attempting_insertion_using_binary_search returns an interval as a list of two floats.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    delta = 0.1

    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
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
    delta = 0.1
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
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

    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    interval = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)

    assert -1<=interval[0] <=1
    assert -1<=interval[1] <=1



#########################################################################################################
# Invalid Input Check for binary search

def test_binary_search_invalid_n_type1():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for negative n.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, -n, m, user_samples)



def test_binary_search_invalid_n_type2():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for n not int.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, n+0.5, m, user_samples)




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
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        attempting_insertion_using_binary_search(-alpha, delta, n, m, user_samples)


def test_binary_search_negative_delta():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for negative delta.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 20
    true_mean = 0.3
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match="delta must be a positive number less than 1"):
        attempting_insertion_using_binary_search(alpha, -delta, n, m, user_samples)

def test_binary_search_invalid_delta_range():
    """
    Check that attempting_insertion_using_binary_search raises ValueError for negative delta.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 1.1
    m = 20
    true_mean = 0.3
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match="delta must be a positive number less than 1"):
        attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)




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
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
    with pytest.raises(ValueError, match=f"user_samples must be a 2D array"):
        attempting_insertion_using_binary_search(alpha, delta, n-1, m, user_samples)



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
    user_samples1,true_mean_scaled = generate_univariate_scaled_data("normal",n1,m, true_mean)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = attempting_insertion_using_binary_search(alpha, delta, n1, m, user_samples1)  
        assert any("is odd" in str(warning.message) for warning in w)
    
def test_binary_search_warns_about_smaller_m():
    """
    Check that a warning is issued when m<7.
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    delta = 0.1
    m = 6
    true_mean = 0.3
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)  
        assert any("is below the recommended minimum 7" in str(warning.message) for warning in w)
    
    
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
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    with pytest.raises(ValueError, match=f"m must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, n, "a", user_samples)


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
    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    with pytest.raises(ValueError, match="m must be a positive integer"):
        attempting_insertion_using_binary_search(alpha, delta, n, -m, user_samples)

    
def test_incorrect_input_rng():
    """
    Check that ValueError is raised if data is not in [-1,1].
    """
    np.random.seed(42)
    alpha = 0.6
    n = 5000
    m = 20
    true_mean = 0.3
    delta=0.1
    user_samples ,true_mean_scaled = generate_univariate_scaled_data("normal",n,m,true_mean)
    samples = 2*user_samples
    with pytest.raises(ValueError, match=f"All entries must lie in"):
        attempting_insertion_using_binary_search(alpha, delta, n, m, samples)



##################################################################################################

def test_binary_search_interval_no_noise():
    """
    Test binary search interval estimation with no noise (alpha = infinity).
    Checks that true mean lies within the estimated interval and the interval length has decreased.
    """
    np.random.seed(42)
    # for no noise alpha = np.inf and pi_alpha = 1
    alpha = np.inf
    n = 9000
    delta = 0.1
    m = 20
    true_mean = 0.3

    user_samples,true_mean_scaled = generate_univariate_scaled_data("normal",n,m, true_mean)
    
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
    n_small = 500
    n_large = 20000
    delta_small=0.1
    delta_large=0.1
    m = 20
    true_mean = 0.3


    user_samples_small,_ = generate_univariate_scaled_data("normal",n_small,m, true_mean)
    user_samples_large,_ = generate_univariate_scaled_data("normal",n_large,m, true_mean)

    L1, R1 = attempting_insertion_using_binary_search(alpha, delta_small, n_small, m, user_samples_small)
    L2, R2 = attempting_insertion_using_binary_search(alpha, delta_large, n_large, m, user_samples_large)

    width1 = R1 - L1
    width2 = R2 - L2

    assert width2 <= width1, f"Interval did not shrink with more users: {width1:.4f} vs {width2:.4f}"


def test_binary_search_constant_data_inf_alpha():
    """
    Tests the case when the we have constant data and there is no noise. 
    """
    n, m = 300, 20
    c = 0.2
    data = [np.full(m, c) for _ in range(n)]
    data = np.asarray(data)
    # No privacy noise i.e. pi_alpha=1
    L, R = attempting_insertion_using_binary_search(np.inf, 0.1, n, m, data)
    # Since there is no noise, we should get the correct interval and interval size should shrink
    assert L <= c <= R
    assert R - L < 1.5  


def test_binary_search_no_iterations():
    # Using small n so t_max is 0
    n, m, alpha, delta = 8, 20, 0.5, 0.1
    user_samples,_ = generate_univariate_scaled_data("normal",n,m, 0.1)
    L, R = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)
    assert (L, R) == (-1.0, 1.0)

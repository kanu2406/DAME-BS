from kent.kent import kent_mean_estimator, partition_interval
import numpy as np
import pytest
import warnings
from experiments.synthetic_data_experiments.univariate_experiment import generate_univariate_scaled_data

#################################################################
########## Test for kent_mean_estimator #########################

def test_ndarray_1d_raises():
    """1D ndarray should raise ValueError about needing a 2D array."""
    np.random.seed(42)
    X = np.zeros(10)
    with pytest.raises(ValueError, match="2D numpy array"):
        kent_mean_estimator(X, alpha=0.5)

def test_ndarray_3d_raises():
    """3D ndarray should raise ValueError about needing a 2D array."""
    np.random.seed(42)
    X = np.zeros((5, 5, 5))
    with pytest.raises(ValueError, match="2D numpy array"):
        kent_mean_estimator(X, alpha=0.5)

def test_non_array_non_list_raises():
    """Passing an integer should raise ValueError about type."""
    np.random.seed(42)
    X = 123
    with pytest.raises(ValueError, match="2D numpy array or a list"):
        kent_mean_estimator(X, alpha=0.5)

def test_empty_list_raises():
    """An empty list has no user samples"""
    np.random.seed(42)
    X = []
    with pytest.raises(ValueError, match="must contain at least one user sample"):
        kent_mean_estimator(X, alpha=0.5)

def test_list_element_not_iterable():
    """If list elements don't support len(), it should error."""
    np.random.seed(42)
    X = [object(), object()]
    with pytest.raises(ValueError, match="Each entry in X must be array-like"):
        kent_mean_estimator(X, alpha=0.5)


def test_list_ragged_lengths():
    """List of arrays with different lengths should raise error."""
    np.random.seed(42)
    X = [np.zeros(3), np.zeros(4)]  # lengths 3 and 4
    with pytest.raises(ValueError, match="must have length 3"):
        kent_mean_estimator(X, alpha=0.5)

def test_negative_alpha():
    """Negative alpha should raise value error."""
    np.random.seed(42)
    X,_ = generate_univariate_scaled_data("normal",1000,10,0.1)
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        kent_mean_estimator(X, alpha=-0.4)

def test_zero_K():
    """K=0 should throw value error."""
    np.random.seed(42)
    X,_ = generate_univariate_scaled_data("normal",1000,10,0.1)
    with pytest.raises(ValueError, match="K must be a positive number"):
        kent_mean_estimator(X, alpha=0.5, K=0)

def test_negative_K():
    """K<0 should throw value error."""
    np.random.seed(42)
    X,_ = generate_univariate_scaled_data("normal",1000,10,0.1)
    with pytest.raises(ValueError, match="K must be a positive number"):
        kent_mean_estimator(X, alpha=0.5, K=-1)

def test_out_of_range_entries():
    """Values of user_samples should lie in [-1,1]"""
    np.random.seed(42)
    X,_ = generate_univariate_scaled_data("normal",1000,10,0.1)
    X_new = 2*X
    with pytest.raises(ValueError, match="All entries of X must lie in"):
        kent_mean_estimator(X_new, alpha=0.5)

def test_output_range_and_type():
    """Estimated mean should lie in [-1,1]"""
    np.random.seed(42)
    X,_ = generate_univariate_scaled_data("normal",1000,10,0.1)
    theta_hat = kent_mean_estimator(X, alpha=0.5)
    assert isinstance(theta_hat, float)
    assert -1 <= theta_hat <= 1

def test_no_noise():
    """
    Test the output when noise is absent (alpha = infinity).
    Checks that estimated mean is within valid range and close to the true mean.
    
    """
    np.random.seed(42)
    X,true_mean_scaled = generate_univariate_scaled_data("normal",1000,10,0.1)
    estimated_mean= kent_mean_estimator(X, alpha=np.inf)
    assert -1 <= estimated_mean <= 1, f"Estimated mean {estimated_mean} not in interval [-1,1]"
    assert np.abs(estimated_mean-true_mean_scaled)**2<= 1e-3, f"Estimated mean and true mean are not close."
    

def test_dame_on_constant_data_inf_alpha():
    """
    Test that on constant data and no noise we should be able to get a very good estimate of estimated mean.
    """
    np.random.seed(0)
    n, m = 1000, 20
    c = 0.1
    user_samples = [np.full(m, c) for _ in range(n)]
    est = kent_mean_estimator(user_samples, alpha=np.inf)
    assert abs(est - c) < 1e-3


def test_small_T_star_path():
    """Choosing alpha small so exp(arg_safe) << T, ensuring T_star < T"""
    
    alpha = 0.1
    X,_ = generate_univariate_scaled_data("normal",20,1000,0.5)
    # Should run without error and produce a float in [-1,1]
    theta_hat = kent_mean_estimator(X, alpha=alpha,K=9)
    assert isinstance(theta_hat, float)
    assert -1 <= theta_hat <= 1


###############################################################################
############## Test for partition_interval ####################################

def test_partition_interval_even_division():
    """with delta = 0.5, we should get exactly two intervals"""
    intervals = partition_interval(-1.0, 1.0, 0.5)
    assert intervals == [(-1.0, 0.0), (0.0, 1.0)]

def test_small_and_large_delta():
    """smaller delta should give more intervals """

    delta1 = 0.1
    delta2 = 0.3
    intervals1 = partition_interval(-1, 1, delta1)
    intervals2 =  partition_interval(-1,1,delta2)
    
    assert len(intervals1)>len(intervals2)

def test_coverage():
    """Testing if union of all intervals is [-1,1]"""

    intervals = partition_interval(-1,1,0.1)

    # We will test that the intervals we got when merged should be equal to [-1,1]
    merged = []
    for L, U in sorted(intervals, key=lambda t: t[0]):
        if not merged:
            # First interval just gets appended
            merged.append([L, U])
        else:
            prev_L, prev_U = merged[-1]
            if L > prev_U:
                # If no overlap or gap, we start a brand‑new interval
                merged.append([L, U])
            else:
                # Extending previous interval's ends
                merged[-1][1] = max(prev_U, U)
    
    
    assert len(merged) == 1
    L0, U0 = merged[0]
    assert pytest.approx(L0) == -1
    assert pytest.approx(U0) == 1
        
    

def test_partition_interval_single():
    """delta = (b-a)/2 should return interval itself"""
    intervals = partition_interval(-1,1,1)
    assert intervals == [(-1,1)]

def test_delta_zero():
    """for delta=0 function should return orignal interval"""
    intervals =  partition_interval(-1,1,0)
    assert intervals == [(-1,1)]
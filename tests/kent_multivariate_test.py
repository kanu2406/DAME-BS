import numpy as np
from experiments.multivariate_experiment import generate_multivariate_scaled_data
from kent.multivariate_kent import kent_multivariate_estimator
import pytest
import warnings

##################################################################################################
# Invalid Input Check 

def test_invalid_user_samples_wrong_dim():
    """
    Check ValueError is raised for wrong dim of user_samples.
    """
    
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(100,10))
    vmin = user_samples.min()
    vmax = user_samples.max()
    user_scaled = 2 * (user_samples - vmin) / (vmax - vmin) - 1
    with pytest.raises(ValueError, match="X must be a 3D numpy array of shape"):
        kent_multivariate_estimator(np.array(user_scaled), alpha=0.5, K=1.0)

def test_values_out_of_range():
    """
    Check ValueError is raised coordinates of user_samples are outside the range [-1,1].
    """
    n, m, d = 10000, 5, 2
    bad = np.ones((n, m, d)) * 2  # all 2 > 1
    with pytest.raises(ValueError, match="All entries of X must lie in"):
        kent_multivariate_estimator(bad, alpha=0.5)


def test_invalid_n():
    """
    Check warning is raised if n is not a multiple of 2d.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 135, 7,4, [0.1,0.1,0.1,0.1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = kent_multivariate_estimator(user_samples, 0.6)
        assert any("Adjusting number of users to the nearest lower multiple of 2d" in str(warning.message) for warning in w)

def test_negative_alpha():
    """
    Check ValueError is raised for negative alpha.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 10000, 10,2, [0.1,0.1])
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        kent_multivariate_estimator(user_samples, -0.6)

def test_invalid_alpha_type():
    """
    Check ValueError is raised if alpha is not float or int.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 10000,10 ,2, [0.1,0.1])
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        kent_multivariate_estimator(user_samples, "a")

def test_negative_K():
    """
    Check ValueError is raised if K<0.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 10000,10 ,2, [0.1,0.1])
    with pytest.raises(ValueError, match="K must be a positive number"):
        kent_multivariate_estimator(user_samples, 0.5,-2)

def test_invalid_K():
    """
    Check ValueError is raised if K is not float or int.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 10000,10 ,2, [0.1,0.1])
    with pytest.raises(ValueError, match="K must be a positive number"):
        kent_multivariate_estimator(user_samples, 0.5,"a")


def test_about_small_n():
    """
    Check that a warning is issued when n<2d.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 3, 10,2, [0.1,0.1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = kent_multivariate_estimator(user_samples, 0.5)  
        assert any("Not enough data for localization or estimation of all coordinates" in str(warning.message) for warning in w)
    assert np.array_equal(result, np.zeros(2))
    

###########################################################################################
# Tests for output ranges
def test_output_range():
    """
    Test that the function returns a d-dimensional array with each coordinate within the valid range [-1, 1].
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 5000, 10,2, [0.1,0.1])
    est_mean=kent_multivariate_estimator(user_samples, 0.5)
    assert est_mean.shape == (2,)
    assert np.all(est_mean >= -1) and np.all(est_mean <= 1) 
    assert isinstance(est_mean, np.ndarray)
    assert est_mean.dtype == float

def test_no_noise():
    """
    Test output when noise is absent (alpha = infinity).
    Checks that estimated mean is within valid range and close to the true mean.
    
    """
    np.random.seed(42)
    user_samples,true_mean_scaled = generate_multivariate_scaled_data("normal", 10000, 20,10, [0.0]*10)
    est_mean=kent_multivariate_estimator(user_samples, np.inf)
    # assert np.allclose(est_mean, true_mean_scaled, atol=1e-4)
    assert np.all(np.abs(est_mean - true_mean_scaled) < 1e-2)



def test_constant_data_exact_recovery():
    """
    Testing the case when data is constant and alpha = inf. We should get a very good estimate of mean.
    """
    np.random.seed(42)
    n, m, d = 8000, 10, 2
    true_vec = np.array([0.1, 0.1])
    user_samples = np.tile(true_vec, (n, m, 1))
    est_mean=kent_multivariate_estimator(user_samples, np.inf)
    assert np.allclose(est_mean, np.array([0.1]*2), atol=1e-8)
    
def test_exact_multiple_of_2d():
    """
    Testing the case when data is exact multiple of 2d then there should be no warnings.
    """
    np.random.seed(42)
    user_samples,_ = generate_multivariate_scaled_data("normal", 5000, 10,2, [0.1,0.1])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est_mean=kent_multivariate_estimator(user_samples, 0.5)
        
    assert not w  # no warnings
    assert est_mean.shape == (2,)




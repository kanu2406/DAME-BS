'''Unit Tests for multivariate_dame_bs_l_inf()'''

import pytest 
from dame_bs.multivariate_dame_bs import multivariate_dame_bs_l_inf
import numpy as np
import warnings

##################################################################################################
# Invalid Input Check 

def test_invalid_user_samples_wrong_dim():
    """
    Check ValueError is raised for wrong dim of user_samples.
    """
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(100,7))
    with pytest.raises(ValueError, match="user_samples must be a 3D numpy array of shape (n, m, d)"):
        multivariate_dame_bs_l_inf(np.array(user_samples), alpha=0.5)

def test_values_out_of_range():
    """
    Check ValueError is raised coordinates of user_samples are outside the range [-1,1].
    """
    n, m, d = 4, 5, 2
    bad = np.ones((n, m, d)) * 2  # all 2 > 1
    with pytest.raises(ValueError, match="All entries must lie in [-1, 1]"):
        multivariate_dame_bs_l_inf(bad, alpha=0.5)



def test_invalid_n():
    """
    Check warning is raised if n is not a multiple of 2d.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(135,7,4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = multivariate_dame_bs_l_inf(user_samples, 0.6)
        assert any("Adjusting number of users to the nearest lower multiple of 2d" in str(warning.message) for warning in w)


def test_negative_alpha():
    """
    Check ValueError is raised for negative alpha.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        multivariate_dame_bs_l_inf(user_samples, alpha=-0.5)

def test_invalid_alpha_type():
    """
    Check ValueError is raised if alpha is not float or int.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        multivariate_dame_bs_l_inf(user_samples, alpha="a")

def test_invalid_user_samples():
    np.random.seed(42)
    with pytest.raises(ValueError, match="alpha must be a positive number"):
        multivariate_dame_bs_l_inf("a", alpha=0.5)



def test_about_smaller_m():
    """
    Check that a warning is issued when m is smaller than 7.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = multivariate_dame_bs_l_inf(user_samples, alpha=0.5)  
        assert any("is below the recommended minimum 7" in str(warning.message) for warning in w)
    
def test_about_small_n():
    """
    Check that a warning is issued when n<2d.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = multivariate_dame_bs_l_inf(user_samples, alpha=0.5)  
        assert any("Not enough data for localization or estimation of all coordinated" in str(warning.message) for warning in w)
    assert result == np.zeros(4)
    




###########################################################################################
# Tests for output ranges
def test_output_range():
    """
    Test that multivariate_dame_bs_l_inf returns a d-dimensional array with each coordinate within the valid range [-1, 1].
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    est_mean=multivariate_dame_bs_l_inf(user_samples, alpha=0.6)
    assert est_mean.shape == (4,)
    assert np.all(est_mean >= -1) and np.all(est_mean <= 1) and isinstance(est_mean, float)

def test_no_noise():
    """
    Test output when noise is absent (alpha = infinity).
    Checks that estimated mean is within valid range and close to the true mean.
    
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(10000,7,4))
    est_mean=multivariate_dame_bs_l_inf(user_samples, alpha=0.6)
    assert np.allclose(est_mean, np.array([0.1]*10), atol=1e-4)


def test_constant_data_exact_recovery():
    """
    Testing the case when data is constant and alpha = inf. We should get a very good estimate of mean.
    """
    np.random.seed(42)
    n, m, d = 8000, 10, 2
    true_vec = np.array([0.1, 0.1])
    user_samples = np.tile(true_vec, (n, m, 1))
    est_mean=multivariate_dame_bs_l_inf(user_samples, alpha=np.inf)
    assert np.allclose(est_mean, np.array([0.1]*10), atol=1e-8)
    
def test_exact_multiple_of_2d():
    """
    Testing the case when data is exact multiple of 2d then there should be no warnings.
    """
    np.random.seed(42)
    user_samples = np.random.normal(loc=0.1, scale=1.0, size=(8000,10,2))
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est_mean=multivariate_dame_bs_l_inf(user_samples, alpha=0.6)
        
    assert not w  # no warnings
    assert est_mean.shape == (2,)




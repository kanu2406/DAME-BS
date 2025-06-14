import pytest 
from dame_ts.ternary_search import attempting_insertion_using_ternary_search
from dame_ts.dame_ts import dame_with_ternary_search
import numpy as np
import math

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


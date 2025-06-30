# DAME-BS: Differentially Private Mean Estimation via Binary Search

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/kanu2406/DAME-BS/tree/main.svg?style=svg&circle-token=CCIPRJ_HwgZxRmn4FC9KWA4t8tmKG_42331c11496c635f99cf9fdd0514727175f5446a)](https://dl.circleci.com/status-badge/redirect/gh/kanu2406/DAME-BS/tree/main) [![codecov](https://codecov.io/gh/kanu2406/DAME-BS/graph/badge.svg?token=LBKGKXDCGV)](https://codecov.io/gh/kanu2406/DAME-BS)

This repository provides a Python implementation of the **DAME-BS** algorithm for locally differentially private mean estimation using binary search. It includes theoretical upper bounds and experiments across multiple distributions.

---

### Requirements

```python
numpy>=1.12
matplotlib>=2.0.0
pytest>=6.2.5
tqdm>=4.62.0
sphinx>=5.3.0
sphinx_rtd_theme>=1.3.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

All dependencies are listed in requirements.txt.


```bash
pip install -r requirements.txt
```

### Installation
Clone the repository:

```bash
git clone https://github.com/kanu2406/DAME-BS.git
cd DAME-TS
```

### Experiments
You can run the included experiments to replicate results:

```python
python experiments/experiment_risk_vs_alpha.py
python experiments/experiment_risk_vs_n.py
```

### Basic Example

```python
import numpy as np
import math
from dame_ts.dame_ts import dame_with_binary_search
from dame_ts.ternary_search import attempting_insertion_using_binary_search
 
alpha = 0.6
pi_alpha = np.exp(alpha) / (1 + np.exp(alpha))
n = 20000
m = 20
true_mean = 0.3
delta = 2 * n * math.exp(-n * (2 * pi_alpha - 1)**2 / 2)

# Generate fake data
user_samples= [np.random.normal(loc=true_mean, scale=0.6, size=m) for _ in range(n)]

# Estimated Interval
L, R = attempting_insertion_using_binary_search(alpha, delta, n, m, user_samples)
print(f"Binary Search Interval: [{L:.3f}, {R:.3f}]")

# Estimated Mean
bar_theta = dame_with_binary_search(n, alpha, m, user_samples)
print(f"Final mean Estimate: {bar_theta:.3f}")



```


### Documentation

The documentation can be found here -

https://kanu2406.github.io/DAME-BS/



### License

This project is licensed under the [MIT License](LICENSE).




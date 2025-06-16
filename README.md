


# DAME-TS: Differentially Private Mean Estimation via Ternary Search

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/kanu2406/DAME-TS/tree/main.svg?style=svg&circle-token=CCIPRJ_HwgZxRmn4FC9KWA4t8tmKG_42331c11496c635f99cf9fdd0514727175f5446a)](https://dl.circleci.com/status-badge/redirect/gh/kanu2406/DAME-TS/tree/main)

This repository provides a Python implementation of the **DAME-TS** algorithm for locally differentially private mean estimation using ternary search. It includes theoretical upper bounds and extensive experiments across multiple distributions.

---

### Requirements

```python
numpy>=1.12
matplotlib>=2.0.0
pytest>=6.2.5
tqdm>=4.62.0
sphinx>=5.3.0
sphinx_rtd_theme>=1.3.0
```

All dependencies are listed in requirements.txt.


```python
pip install -r requirements.txt
```

### Installation
Clone the repository:

```python
git clone https://github.com/kanu2406/DAME-TS.git
```

### Experiments
You can run the included experiments to replicate results:

```python
python experiments/experiment_risk_vs_alpha.py
python experiments/experiment_risk_vs_n.py
```

### Documentation

The documentation can be found here -

https://kanu2406.github.io/DAME-TS/



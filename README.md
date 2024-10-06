# fast-median

Median estimator

Python implementation of real time median estimator

# Publication

For an explanation of the algorithm see

Burman, Solé-Casals, Lew (2024) Robust and Memory-less Median Estimation for Real-Time Spike Detection

Extracellular recording used to validate the estimator

DOI: [10.5281/zenodo.13895772](https://doi.org/10.5281/zenodo.13895772)

# Installation

Create a virtual environment

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e .
```

# Tests

```
python -m median.tests.test_new_median
python -m median.tests.test_new_median_int
```

# Development and plots

```
pip install -r requirements.dev.txt
```

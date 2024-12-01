# Fast Median

Median estimator

Python implementation of real time median estimator

## Publication

For an explanation of the algorithm see [1].

If you use this algorithm, pleas cite:

Burman A, Solé-Casals J, Lew SE (2024) Robust and memory-less median estimation for real-time spike detection. PLoS ONE 19(11): e0308125. [10.1371/journal.pone.0308125](https://doi.org/10.1371/journal.pone.0308125)

Extracellular recording used to validate the estimator

DOI: [10.5281/zenodo.13895772](https://doi.org/10.5281/zenodo.13895772)

## Installation

Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e .
```

### Tests

```bash
python -m median.tests.test_new_median
python -m median.tests.test_new_median_int
```

### Development and plots

```bash
pip install -r requirements.dev.txt
```

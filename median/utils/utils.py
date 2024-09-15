from math import floor

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt


def smooth(data, W=10):
    return filtfilt(np.ones(W) / W, 1, data)


def getBinsX(minbin, maxbin, nbins=101, step=None):
    if step:
        nbins = int((maxbin - minbin) / step) + 1
        bins = np.linspace(minbin - step / 2, maxbin + step / 2, nbins + 1)
        xp = np.linspace(minbin, maxbin, nbins, endpoint=True)
    else:
        step = (maxbin - minbin) / (nbins - 1)
        bins = np.linspace(minbin - step / 2, maxbin + step / 2, nbins + 1)
        xp = np.linspace(minbin, maxbin, nbins, endpoint=True)

    return bins, xp


def sign_test(samp, mu0=0):
    samp = np.asarray(samp)
    pos = np.sum(samp > mu0)
    neg = np.sum(samp < mu0)
    M = (pos - neg) / 2.0
    p = stats.binomtest(min(pos, neg), pos + neg, 0.5)
    return M, p


def highPassFilter(signal, Fs, High):
    b, a = butter(N=3, Wn=(High / (Fs / 2)), btype="highpass")
    data = filtfilt(b, a, signal)
    return data


def bandPassFilter(signal, Fs, Low, High):
    b, a = butter(N=3, Wn=[Low / (Fs / 2), High / (Fs / 2)], btype="bandpass")
    data = filtfilt(b, a, signal)
    return data


def distance_buffer(L):
    """
    L: lenght buffer, odd
    """
    assert L % 2 == 1
    m = int((L + 1) / 2)
    buff = np.square(np.linspace(1, L, L, dtype=int) - m)
    buff[m - 1] = 1
    return buff


def distance(i, L=63):
    d = (i - (L + 1) / 2) ** 2
    if d:
        return d
    else:
        return 1


def energy(buff, median, weight):
    """
    buff = numpy array odd lenght
    median: int real median value
    """
    e = np.dot(np.square(buff - median), weight)
    return e


def run_classic_median(data: np.ndarray, L: int = 63):
    assert data.ndim == 1
    buff = data[:L].copy()
    N = data.shape[0]
    a = np.empty_like(data)
    for i in range(N):
        np.concatenate((buff[1:], [data[i]]), out=buff)
        a[i] = np.median(buff)

    return a, buff


def run_classic_mean(data: np.ndarray, L: int = 63):
    assert data.ndim == 1
    buff = data[:L]
    N = data.shape[0] - L
    a = np.empty_like(data)
    for i in range(N):
        np.concatenate((buff[1:], [data[i + L]]), out=buff)
        a[i] = np.mean(buff)

    return a


def pretty_time(total_time: float) -> str:
    """returns a line to print prettier time past"""
    return f"total --- {int(floor(total_time) / 60):d} minutes, {total_time % 60:.2f} seconds ---"

from typing import cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from median.distributions import AbsNormal, Beta, LogNormal, Normal, Uniform
from median.utils.utils import getBinsX


def test_Uniform():
    L = 40
    a = 20
    b = 120
    dist = Uniform()

    x = np.linspace(a - 20, b + 20, 1500)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(Uniform(a, b, L))
    data = dist.samples(100000, a, b)
    ax[0][0].plot(x, dist.pdf(x, a, b))
    bins, _ = getBinsX(0, 140, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x, a, b))
    ax[2][0].plot(x, dist.cdf_low(x, (a + b) / 2, a, b))
    ax[3][0].plot(x, dist.cdf_high(x, (a + b) / 2, a, b))
    ax[0][1].plot(x, dist.pdf_min(x, a, b, L))
    ax[1][1].plot(x, dist.cdf_min(x, a, b, L))
    ax[2][1].plot(x, dist.pdf_max(x, a, b, L))
    ax[3][1].plot(x, dist.cdf_max(x, a, b, L))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()

    dist = Uniform(5, 10, 5)
    x = np.linspace(0, 15, 1500)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(100000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(0, 15, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 6.5))
    ax[3][0].plot(x, dist.cdf_high(x, 6.5))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()

    dist.cdf(7.5)
    dist.cdf([7.5, 8.5])
    dist.cdf_low(5, 7)
    dist.cdf_low([5, 5.5], 7)
    dist.cdf_high(7, 6)
    dist.cdf_high([7, 7.3], 6)
    dist.pdf_min(5.5)
    dist.pdf_min([5.5, 5.3])
    dist.cdf_min(5.5)
    dist.cdf_min([5.5, 5.3])
    dist.pdf_max(9.5)
    dist.pdf_max([9.5, 9.7])
    dist.cdf_max(9.5)
    dist.cdf_max([9.5, 9.7])


def test_Normal():
    dist = Normal(0, 2, 55)
    x = np.linspace(-10, 10, 1500)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(100000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(-10, 10, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 1))
    ax[3][0].plot(x, dist.cdf_high(x, 1))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()

    dist = Normal(20, 2, 55)
    x = np.linspace(10, 30, 1500)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(100000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(10, 30, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 21))
    ax[3][0].plot(x, dist.cdf_high(x, 21))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()


def test_Abs_Normal():
    dist = AbsNormal(2, 7)
    x = np.linspace(-1, 20, 10000)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(100000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(0, 15, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 1))
    ax[3][0].plot(x, dist.cdf_high(x, 3))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()


def test_Beta():
    dist = Beta(2, 5, 63)
    x = np.linspace(-0.1, 1.1, 10000)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(1000000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(0, 1, nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 0.1))
    ax[3][0].plot(x, dist.cdf_high(x, 0.3))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()


def test_LogNormal():
    dist = LogNormal(0, 0.4, 15)
    x = np.linspace(-0.1, 5, 1500)
    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = cast(list[list[Axes]], ax)

    fig.suptitle(dist)
    data = dist.samples(1000000)
    ax[0][0].plot(x, dist.pdf(x))
    bins, _ = getBinsX(0, x[-1], nbins=100)
    ax[0][0].hist(data, bins=bins, density=1)
    ax[1][0].plot(x, dist.cdf(x))
    ax[2][0].plot(x, dist.cdf_low(x, 1.5))
    ax[3][0].plot(x, dist.cdf_high(x, 2))
    ax[0][1].plot(x, dist.pdf_min(x))
    ax[1][1].plot(x, dist.cdf_min(x))
    ax[2][1].plot(x, dist.pdf_max(x))
    ax[3][1].plot(x, dist.cdf_max(x))

    fig.set_size_inches(10, 8)
    ax[0][0].set_title("pdf")
    ax[1][0].set_title("cdf")
    ax[2][0].set_title("cdf_low")
    ax[3][0].set_title("cdf_high")
    ax[0][1].set_title("pdf min")
    ax[1][1].set_title("cdf min")
    ax[2][1].set_title("pdf max")
    ax[3][1].set_title("cdf max")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    test_Uniform()
    test_Normal()
    test_Abs_Normal()
    test_Beta()
    test_LogNormal()

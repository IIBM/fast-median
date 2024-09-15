import numpy as np
import scipy.integrate as integrate
from scipy.special import beta as betaf  # pylint: disable=no-name-in-module
from scipy.special import btdtri  # pylint: disable=no-name-in-module
from scipy.special import btdtr, erf, erfinv


class Distribution:
    "Template for distributions"

    def __init__(self):
        pass

    def cfd(self, x):
        """
        cumulative probability density function
        cdf = int_-inf^x pdf
        """

    def pdf(self, x):
        """
        probability density function
        pdf = d/dx  cdf
        """

    def cfd_low(self, x, x0):
        """
        cdf given that we know x<x0)
        int -inf and x of pdf(x|x<xo)
        """

    def cfd_high(self, x, x0):
        """
        cdf given that we know x>x0
        int -inf and x of pdf(x|x>xo)
        """

    def cdf_max(self, x, L=1):
        """
        Having L samples, the cdf distribution of the maximum sample
        """

    def cdf_min(self, x, L=1):
        """
        Having L samples, the cdf distribution of the minimun sample
        """

    def pdf_max(self, x, L=1):
        """
        d/dx cdf_max
        """

    def pdf_min(self, x, L=1):
        """
        d/dx cdf_min
        """

    def mean_min(self, L):
        """ """

    def mean_max(self, L):
        """ """

    def samples(self, N):
        """ """

    def mean(self) -> float:
        """ """

    def median(self) -> float:
        """ """

    def stddev(self) -> float:
        """ """

    def __str__(self) -> str:
        """
        Info to represent self
        """


class Uniform(Distribution):
    def __init__(self, xmin=0, xmax=1, L=1):
        self._xmin = xmin
        self._xmax = xmax
        self._L = L
        self._mean_min = np.nan
        self._mean_max = np.nan

    def cdf(self, x, xmin=None, xmax=None):
        """
        x should be a 1 dimension numpy array
        """
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        xarr = np.array(x)
        if xarr.shape == ():
            out = 0 if x < xmin else 1 if x > xmax else (x - xmin) / (xmax - xmin)
        else:
            out = (xarr - xmin) / (xmax - xmin)
            out[np.where(xarr < xmin)] = 0
            out[np.where(xarr > xmax)] = 1
        return out

    def cdf_low(self, x, x0, xmin=None, xmax=None):
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        xarr = np.array(x)
        if x0 > xmax:
            out = self.cdf(x, xmin, xmax)
        elif x0 < xmin:
            out = 0 if xarr.shape == () else np.zeros_like(xarr)
        else:
            out = self.cdf(x, xmin, xmax) / self.cdf(x0, xmin, xmax)
            if xarr.shape != ():
                out[np.where(xarr > x0)] = 1
        return out

    def cdf_high(self, x, x0, xmin=None, xmax=None):
        """
        x should be a 1 dimension numpy array
        """
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        xarr = np.array(x)
        if x0 > xmax:
            out = 0 if xarr.shape == () else np.zeros_like(x)
        elif x0 < xmin:
            out = self.cdf(x, xmin, xmax)
        else:
            out = (self.cdf(x, xmin, xmax) - self.cdf(x0, xmin, xmax)) / (
                1 - self.cdf(x0, xmin, xmax)
            )
            if xarr.shape != ():
                out[np.where(xarr < x0)] = 0
        return out

    def pdf(self, x, xmin=None, xmax=None):
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        xarr = np.array(x)
        if xarr.shape == ():
            out = 0 if (x < xmin or x > xmax) else 1 / (xmax - xmin)
        else:
            out = np.ones_like(xarr) / (xmax - xmin)
            out[np.where(xarr < xmin)] = 0
            out[np.where(xarr > xmax)] = 0
        return out

    def cdf_max(self, x, L=None, xmin=None, xmax=None):
        L = L if L else self._L

        return (self.cdf(x, xmin, xmax)) ** L

    def pdf_max(self, x, L=None, xmin=None, xmax=None):
        L = L if L else self._L

        return L * (self.cdf(x, xmin, xmax)) ** (L - 1) * self.pdf(x, xmin, xmax)

    def mean_max(self, L=None, xmin=None, xmax=None):
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        if np.isnan(self._mean_max):
            result = integrate.quad(
                lambda x: x * self.pdf_max(x, L, xmin, xmax), xmin, xmax
            )
            if result[1] / result[0] < 1e-5:
                self._mean_max = result[0]
            else:
                print("warning: mean_max could not be calculated")

        return self._mean_max

    def cdf_min(self, x, L=None, xmin=None, xmax=None):
        L = L if L else self._L

        return 1 - (1 - self.cdf(x, xmin, xmax)) ** L

    def pdf_min(self, x, L=None, xmin=None, xmax=None):
        L = L if L else self._L

        return L * (1 - self.cdf(x, xmin, xmax)) ** (L - 1) * self.pdf(x, xmin, xmax)

    def mean_min(self, L=None, xmin=None, xmax=None):
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        if np.isnan(self._mean_min):
            result = integrate.quad(
                lambda x: x * self.pdf_min(x, L, xmin, xmax), xmin, xmax
            )
            if result[1] / result[0] < 1e-5:
                self._mean_min = result[0]
            else:
                print("warning: mean_min could not be calculated")

        return self._mean_min

    def samples(self, N=None, xmin=None, xmax=None):
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        if N:
            return np.random.rand(N) * (xmax - xmin) + xmin
        else:
            return np.random.rand() * (xmax - xmin) + xmin

    def mean(self, xmin=None, xmax=None) -> float:
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        return (xmin + xmax) / 2

    def median(self, xmin=None, xmax=None) -> float:
        return self.mean(xmin, xmax)

    def stddev(self, xmin=None, xmax=None) -> float:
        xmin = xmin if xmin else self._xmin
        xmax = xmax if xmax else self._xmax

        return (xmax - xmin) / np.sqrt(12)

    def __str__(self) -> str:
        msg = f"Uniform distribution: xmin={self._xmin:3.2f}, "
        msg += f"xmax={self._xmax:3.2f}, L={self._L:d}, "
        msg += f"mean_min={self.mean_min():3.2f}, mean_max={self.mean_max():3.2f}"

        return msg


class Normal(Distribution):
    def __init__(self, mu=0, sigma=1, L=1):
        self._mu = mu
        self._sigma = sigma
        self._L = L
        self._mean_min = np.nan
        self._mean_max = np.nan

    def pdf(self, x, mu=None, sigma=None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        return (
            1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)
        )

    def cdf(self, x, mu=None, sigma=None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        return 1 / 2 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    def cdf_low(self, x, x0, mu=None, sigma=None):
        out = self.cdf(x, mu, sigma) / self.cdf(x0, mu, sigma)
        if np.array(x).shape == ():
            out = 0 if x < x0 else out
        else:
            out[np.where(x > x0)] = 1
        return out

    def cdf_high(self, x, x0, mu=None, sigma=None):
        out = (self.cdf(x, mu, sigma) - self.cdf(x0, mu, sigma)) / (
            1 - self.cdf(x0, mu, sigma)
        )
        if np.array(x).shape == ():
            out = 0 if x < x0 else out
        else:
            out[np.where(x < x0)] = 0
        return out

    def cdf_max(self, x, L=None, mu=None, sigma=None):
        L = L if L else self._L

        return (self.cdf(x, mu, sigma)) ** L

    def pdf_max(self, x, L=None, mu=None, sigma=None):
        L = L if L else self._L

        return L * (self.cdf(x, mu, sigma)) ** (L - 1) * self.pdf(x, mu, sigma)

    def mean_max(self, L=None, mu=None, sigma=None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_max):
            result = integrate.quad(
                lambda x: x * self.pdf_max(x, L, mu, sigma), mu, mu + 15 * sigma
            )
            if result[1] / result[0] < 1e-5:
                self._mean_max = result[0]
            else:
                print("warning: mean_max could not be calculated")

        return self._mean_max

    def cdf_min(self, x, L=None, mu=None, sigma=None):
        L = L if L else self._L

        return 1 - (1 - self.cdf(x, mu, sigma)) ** L

    def pdf_min(self, x, L=None, mu=None, sigma=None):
        L = L if L else self._L

        return L * (1 - self.cdf(x, mu, sigma)) ** (L - 1) * self.pdf(x, mu, sigma)

    def mean_min(self, L=None, mu=None, sigma=None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_min):
            result = integrate.quad(
                lambda x: x * self.pdf_min(x, L, mu, sigma), -mu - 15 * sigma, mu
            )
            if result[1] / result[0] < 1e-5:
                self._mean_min = result[0]
            else:
                print("warning: mean_min could not be calculated")

        return self._mean_min

    def samples(self, N=None, mu=None, sigma=None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if N:
            return np.random.randn(N) * sigma + mu
        else:
            return np.random.randn() * sigma + mu

    def mean(self, mu=None) -> float:
        mu = mu if mu else self._mu

        return mu

    def median(self, mu=None) -> float:
        return self.mean(mu)

    def stddev(self, sigma=None) -> float:
        sigma = sigma if sigma else self._sigma

        return sigma

    def __str__(self) -> str:
        msg = f"Normal distribution: mu={self._mu:3.2f}, sigma={self._sigma:3.2f}, "
        msg += f"L={self._L:d}, mean_min={self.mean_min():3.2f}, mean_max={self.mean_max():3.2f}"
        return msg


class AbsNormal(Distribution):
    def __init__(self, sigma=1, L=1):
        self._sigma = sigma
        self._L = L
        self._mean_min = np.nan
        self._mean_max = np.nan

    def pdf(self, x, sigma=None):
        sigma = sigma if sigma else self._sigma

        out = 2 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (x / sigma) ** 2)
        if np.array(x).shape == ():
            out = 0 if x < 0 else out
        else:
            out[np.where(x < 0)] = 0
        return out

    def cdf(self, x, sigma=None):
        sigma = sigma if sigma else self._sigma

        out = erf(x / (sigma * np.sqrt(2)))
        if np.array(x).shape == ():
            out = 0 if x < 0 else out
        else:
            out[np.where(x < 0)] = 0
        return out

    def cdf_low(self, x, x0, sigma=None):
        out = self.cdf(x, sigma) / self.cdf(x0, sigma)
        if np.array(x).shape == ():
            out = 0 if x < x0 else out
        else:
            out[np.where(x > x0)] = 1
        return out

    def cdf_high(self, x, x0, sigma=None):
        out = (self.cdf(x, sigma) - self.cdf(x0, sigma)) / (1 - self.cdf(x0, sigma))
        if np.array(x).shape == ():
            out = 0 if x < x0 else out
        else:
            out[np.where(x < x0)] = 0
        return out

    def cdf_max(self, x, L=None, sigma=None):
        L = L if L else self._L

        return (self.cdf(x, sigma)) ** L

    def pdf_max(self, x, L=None, sigma=None):
        L = L if L else self._L

        return L * (self.cdf(x, sigma)) ** (L - 1) * self.pdf(x, sigma)

    def mean_max(self, L=None, sigma=None):
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_max):
            result = integrate.quad(
                lambda x: x * self.pdf_max(x, L, sigma), 0, 10 * sigma
            )
            if result[1] / result[0] < 1e-5:
                self._mean_max = result[0]
            else:
                print("warning: mean_max could not be calculated")

        return self._mean_max

    def cdf_min(self, x, L=None, sigma=None):
        L = L if L else self._L

        return 1 - (1 - self.cdf(x, sigma)) ** L

    def pdf_min(self, x, L=None, sigma=None):
        L = L if L else self._L

        return L * (1 - self.cdf(x, sigma)) ** (L - 1) * self.pdf(x, sigma)

    def mean_min(self, L=None, sigma=None):
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_min):
            result = integrate.quad(
                lambda x: x * self.pdf_min(x, L, sigma), 0, 5 * sigma
            )
            if result[1] / result[0] < 1e-5:
                self._mean_min = result[0]
            else:
                print("warning: mean_min could not be calculated")

        return self._mean_min

    def samples(self, N=None, sigma=None):
        sigma = sigma if sigma else self._sigma

        if N:
            return np.abs(np.random.randn(N) * sigma)
        else:
            return np.abs(np.random.randn() * sigma)

    def mean(self, sigma=None) -> float:
        sigma = sigma if sigma else self._sigma

        return sigma * np.sqrt(2 / np.pi)

    def median(self, sigma=None) -> float:
        sigma = sigma if sigma else self._sigma

        # the cdf of a normal distribution with mu=0 has to give 0.75
        return sigma * np.sqrt(2) * erfinv(2 * 0.75 - 1)

    def stddev(self, sigma=None) -> float:
        sigma = sigma if sigma else self._sigma

        return sigma * np.sqrt(1 - 2 / np.pi)

    def __str__(self):
        msg = f"AbsNormal distribution: sigma={self._sigma:3.2f}, L={self._L:d}, "
        msg += f"mean_min={self.mean_min():3.2f}, mean_max={self.mean_max():3.2f}"
        return msg


class Beta(Distribution):
    def __init__(self, alpha, beta, L=1):
        self._alpha = alpha
        self._beta = beta
        self._L = L
        self._mean_min = np.nan
        self._mean_max = np.nan

    def pdf(self, x, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        if np.array(x).shape == ():
            out = (
                np.power(x, (alpha - 1))
                * np.power((1 - x), (beta - 1))
                / betaf(alpha, beta)
                if 0 < x < 1
                else 0
            )
        else:
            out = [
                (
                    (
                        np.power(i, (alpha - 1))
                        * np.power((1 - i), (beta - 1))
                        / betaf(alpha, beta)
                    )
                    if 0 < i < 1
                    else 0
                )
                for i in x
            ]
        return out

    def cdf(self, x, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        if np.array(x).shape == ():
            out = 1 if x > 1 else 0 if x < 0 else btdtr(alpha, beta, x)
        else:
            out = btdtr(alpha, beta, x)
            out[np.where(x > 1)] = 1
            out[np.where(x < 0)] = 0
        return out

    def cdf_low(self, x, x0, alpha=None, beta=None):
        out = self.cdf(x, alpha, beta) / self.cdf(x0, alpha, beta)
        if np.array(x).shape == ():
            out = 1 if x > x0 else out
        else:
            out[np.where(x > x0)] = 1
        return out

    def cdf_high(self, x, x0, alpha=None, beta=None):
        out = (self.cdf(x, alpha, beta) - self.cdf(x0, alpha, beta)) / (
            1 - self.cdf(x0, alpha, beta)
        )
        if np.array(x).shape == ():
            out = 0 if x < x0 else out
        else:
            out[np.where(x < x0)] = 0
        return out

    def cdf_max(self, x, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        return (self.cdf(x, alpha, beta)) ** L

    def pdf_max(self, x, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        return L * (self.cdf(x, alpha, beta)) ** (L - 1) * self.pdf(x, alpha, beta)

    def mean_max(self, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        if np.isnan(self._mean_max):
            result = integrate.quad(lambda x: x * self.pdf_max(x, L, alpha, beta), 0, 1)
            if result[1] / result[0] < 1e-5:
                self._mean_max = result[0]
            else:
                print("warning: mean_max could not be calculated")

        return self._mean_max

    def cdf_min(self, x, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        return 1 - (1 - self.cdf(x, alpha, beta)) ** L

    def pdf_min(self, x, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        return L * (1 - self.cdf(x, alpha, beta)) ** (L - 1) * self.pdf(x, alpha, beta)

    def mean_min(self, L=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta
        L = L if L else self._L

        if np.isnan(self._mean_min):
            result = integrate.quad(lambda x: x * self.pdf_min(x, L, alpha, beta), 0, 1)
            if result[1] / result[0] < 1e-5:
                self._mean_min = result[0]
            else:
                print("warning: mean_min could not be calculated")

        return self._mean_min

    def samples(self, N=None, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        if N:
            return btdtri(alpha, beta, np.random.rand(N))
        else:
            return btdtri(alpha, beta, np.random.rand())

    def mean(self, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        return alpha / (alpha + beta)

    def median(self, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        return btdtri(alpha, beta, 0.5)

    def stddev(self, alpha=None, beta=None):
        alpha = alpha if alpha else self._alpha
        beta = beta if beta else self._beta

        return np.sqrt(alpha * beta / (alpha + beta + 1)) / (alpha + beta)

    def __str__(self) -> str:
        msg = f"Beta distribution: alpha={self._alpha:3.2f}, "
        msg += f"beta={self._beta:3.2f}, L={self._L:d}, "
        msg += f"meadian={self.median():3.2f}, mean={self.mean():3.2f}, "
        msg += f"mean_min={self.mean_min():3.2f}, mean_max={self.mean_max():3.2f}"
        return msg


class LogNormal(Distribution):
    def __init__(self, mu: float = 0, sigma: float = 1, L: int = 1):
        self._mu = mu
        self._sigma = sigma
        assert self._sigma > 0
        self._L = L
        self._mean_min = np.nan
        self._mean_max = np.nan

    def pdf(self, x, mu: float | None = None, sigma: float | None = None):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.array(x).shape == ():
            out = (
                (
                    1
                    / (x * sigma * np.sqrt(2 * np.pi))
                    * np.exp(-1 / 2 * ((np.log(x) - mu) / sigma) ** 2)
                )
                if x >= 0
                else 0
            )
        else:
            out = np.zeros_like(x)
            index = np.where(x > 0)
            out[index] = (
                1
                / (x[index] * sigma * np.sqrt(2 * np.pi))
                * np.exp(-1 / 2 * ((np.log(x[index]) - mu) / sigma) ** 2)
            )
        return out

    def cdf(
        self, x: float | np.ndarray, mu: float | None = None, sigma: float | None = None
    ):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.array(x).shape == ():
            out = (
                1 / 2 * (1 + erf((np.log(x) - mu) / (sigma * np.sqrt(2))))
                if x > 0
                else 0
            )
        else:
            out = np.zeros_like(x)
            index = np.where(x > 0)
            out[index] = (
                1 / 2 * (1 + erf((np.log(x[index]) - mu) / (sigma * np.sqrt(2))))
            )
        return out

    def cdf_low(
        self,
        x: float | np.ndarray,
        x0: float,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        if np.array(x).shape == ():
            out = self.cdf(x, mu, sigma) / self.cdf(x0, mu, sigma) if x >= x0 else 0
        else:
            out = np.zeros_like(x)
            index = np.where(x > 0)
            out[index] = self.cdf(x[index], mu, sigma) / self.cdf(x0, mu, sigma)
            out[np.where(x > x0)] = 1
        return out

    def cdf_high(
        self,
        x: float | np.ndarray,
        x0: float,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        out = (self.cdf(x, mu, sigma) - self.cdf(x0, mu, sigma)) / (
            1 - self.cdf(x0, mu, sigma)
        )
        if np.array(x).shape == ():
            out = (
                (self.cdf(x, mu, sigma) - self.cdf(x0, mu, sigma))
                / (1 - self.cdf(x0, mu, sigma))
                if x >= x0
                else 0
            )
        else:
            out = np.zeros_like(x)
            index = np.where(x >= x0)
            out[index] = (self.cdf(x[index], mu, sigma) - self.cdf(x0, mu, sigma)) / (
                1 - self.cdf(x0, mu, sigma)
            )
        return out

    def cdf_max(
        self,
        x: float | np.ndarray,
        L: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        L = L if L else self._L

        return (self.cdf(x, mu, sigma)) ** L

    def pdf_max(
        self,
        x: float | np.ndarray,
        L: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        L = L if L else self._L

        return L * (self.cdf(x, mu, sigma)) ** (L - 1) * self.pdf(x, mu, sigma)

    def mean_max(
        self, L: int | None = None, mu: float | None = None, sigma: float | None = None
    ):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_max):
            result = integrate.quad(
                lambda x: x * self.pdf_max(x, L, mu, sigma), mu, mu + 15 * sigma
            )
            if result[1] / result[0] < 1e-5:
                self._mean_max = result[0]
            else:
                print("warning: mean_max could not be calculated")

        return self._mean_max

    def cdf_min(
        self,
        x,
        L: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        L = L if L else self._L

        return 1 - (1 - self.cdf(x, mu, sigma)) ** L

    def pdf_min(
        self,
        x,
        L: int | None = None,
        mu: float | None = None,
        sigma: float | None = None,
    ):
        L = L if L else self._L

        return L * (1 - self.cdf(x, mu, sigma)) ** (L - 1) * self.pdf(x, mu, sigma)

    def mean_min(
        self, L: int | None = None, mu: float | None = None, sigma: float | None = None
    ):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if np.isnan(self._mean_min):
            result = integrate.quad(
                lambda x: x * self.pdf_min(x, L, mu, sigma), 0, sigma
            )
            if result[1] / result[0] < 1e-5:
                self._mean_min = result[0]
            else:
                print("warning: mean_min could not be calculated")

        return self._mean_min

    def samples(
        self, N: int | None = None, mu: float | None = None, sigma: float | None = None
    ):
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        if N:
            return np.random.lognormal(mean=mu, sigma=sigma, size=N)
        else:
            return np.random.lognormal(mean=mu, sigma=sigma)

    def mean(self, mu: float | None = None, sigma: float | None = None) -> float:
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        return np.exp(mu + sigma**2 / 2)

    def median(self, mu: float | None = None) -> float:
        mu = mu if mu else self._mu

        return mu

    def stddev(self, mu: float | None = None, sigma: float | None = None) -> float:
        mu = mu if mu else self._mu
        sigma = sigma if sigma else self._sigma

        return np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)

    def __str__(self) -> str:
        msg = f"Log Normal distribution: mu={self._mu:3.2f}, sigma={self._sigma:3.2f}, "
        msg += f"L={self._L:d}, mean_min={self.mean_min():3.2f}, mean_max={self.mean_max():3.2f}"
        return msg

"""
New median implementation
"""

import numpy as np

from .buffer import Buffer
from .utils.utils import distance_buffer


class Median(Buffer):
    """New median"""

    def __init__(self, L=62, dtype=float, initB=None):
        Buffer.__init__(self, L=L, dtype=dtype, initB=initB)

    def update(self, new_value: float | int):
        # if newValue is bigger than the median, leftmost item will be dropped,
        # if newValue is lower than the median, rightmost item will be dropped,
        # if newValue is equal to the median, its random choice
        # newValue is always inserted into the buffer and in a sorted position
        new_data = np.asarray(new_value, self._dtype)

        if new_data > self._buffer[self._Midx]:
            idx = np.searchsorted(self._buffer, new_data)
            np.concatenate(
                (self._buffer[1:idx], [new_data], self._buffer[idx:]),
                out=self._buffer,
            )
        elif new_data < self._buffer[self._Midx]:
            idx = np.searchsorted(self._buffer, new_data)
            np.concatenate(
                (self._buffer[: idx + 1], self._buffer[idx:-1]),
                out=self._buffer,
            )
            self._buffer[idx] = new_data
        else:  # new_data[0] == self._buffer[self._Midx]:
            idx = self._Midx
            if np.random.default_rng().uniform(size=1) > 0.5:
                np.concatenate(
                    (self._buffer[1:idx], [new_data], self._buffer[idx:]),
                    out=self._buffer,
                )
            else:
                np.concatenate(
                    (self._buffer[: idx + 1], self._buffer[idx:-1]),
                    out=self._buffer,
                )
                self._buffer[idx] = new_data

    def updateR(self, new_value: float | int):
        # same as update, also returns de discarded value
        new_data = np.asarray(new_value, dtype=self._dtype)

        if new_data > self._buffer[self._Midx]:
            idx = np.searchsorted(self._buffer, new_data)
            drop = self._buffer[0]
            np.concatenate(
                (self._buffer[1:idx], [new_data], self._buffer[idx:]),
                out=self._buffer,
            )
        elif new_data < self._buffer[self._Midx]:
            idx = np.searchsorted(self._buffer, new_data)
            drop = self._buffer[-1]
            np.concatenate(
                (self._buffer[: idx + 1], self._buffer[idx:-1]),
                out=self._buffer,
            )
            self._buffer[idx] = new_data
        else:  # new_data[0] == self._buffer[self._Midx]:
            idx = self._Midx
            if np.random.default_rng().uniform(size=1) > 0.5:
                drop = self._buffer[0]
                np.concatenate(
                    (self._buffer[1:idx], [new_data], self._buffer[idx:]),
                    out=self._buffer,
                )
            else:
                drop = self._buffer[-1]
                np.concatenate(
                    (self._buffer[: idx + 1], self._buffer[idx:-1]),
                    out=self._buffer,
                )
                self._buffer[idx] = new_data
        return drop

    def update_inef(self, new_value: float | int):
        # same as update, not efficient, but more legible

        new_data = np.asarray(new_value, dtype=self._dtype)
        # introduce the new value and drop one of the tails
        if new_data > self._buffer[self._Midx]:
            idx = self._L
            for k in range(self._Midx + 1, self._L):
                if new_data < self._buffer[k]:
                    idx = k
                    break
            newBuffer = np.concatenate(
                (self._buffer[1:idx], np.array([new_data]), self._buffer[idx:])
            )
        elif new_data < self._buffer[self._Midx]:
            idx = -1
            for k in range(self._Midx - 1, -1, -1):
                if new_data > self._buffer[k]:
                    idx = k
                    break
            newBuffer = np.concatenate(
                (
                    self._buffer[: idx + 1],
                    np.array([new_data]),
                    self._buffer[idx + 1 : -1],
                )
            )
        elif new_data == self._buffer[self._Midx]:
            if np.random.default_rng().uniform(size=1) > 0.5:
                idx = self._Midx + 1
                newBuffer = np.concatenate(
                    (self._buffer[1:idx], np.array([new_data]), self._buffer[idx:])
                )
            else:
                idx = self._Midx
                newBuffer = np.concatenate(
                    (self._buffer[:idx], np.array([new_data]), self._buffer[idx:-1])
                )
        else:
            raise ValueError("Error in new_data")

        self._buffer = newBuffer

    def get_energy(self, median, weights=None):
        if isinstance(weights, type(None)):
            weights = distance_buffer(self._L)
        e = np.dot(np.square(self._buffer - median), weights)
        return e


def run_new_median(data: np.ndarray, L: int = 63) -> tuple[np.ndarray, np.ndarray]:
    """
    data: Array 1D
    L: Buffer length

    Returns:
     - Array 1D: output at each time step
     - Buffer at the last time step
    """
    assert data.ndim == 1
    buff = Median(L, dtype=data.dtype, initB=data[:L])
    N = data.shape[0]
    a = np.empty_like(data)
    for i in range(N):
        buff.update(data[i])
        a[i] = buff.getMedian()
    return a, buff.getBuffer()


def run_new_median_full(
    data: np.ndarray, store: np.ndarray, L: int = 63, idxs: list[int] = None
) -> None:
    """
    data: Array 1D
    store: Array 2D N x L where the buffer is going to be stored for every step time
    L: Buffer length
    """
    assert data.ndim == 1
    D = data.shape[0] - L
    N = D if idxs is None else len(idxs)
    assert store.shape == (N, L)
    buff = Median(L, dtype=data.dtype, initB=data[:L])
    for i in range(D):
        buff.update(data[i + L])
        if i in idxs:
            store[np.where(idxs == i)[0][0]] = buff.getBuffer()

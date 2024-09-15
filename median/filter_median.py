"""
Filter Median Class
"""

import numpy as np

from .buffer import Buffer


class FilterMedian(Buffer):
    def __init__(self, L=62, dtype=float, initB=None):
        Buffer.__init__(self, L=L, dtype=dtype, initB=initB)
        self._buffer = self._buffer[::-1]
        self._history = np.linspace(0, L - 1, num=L, dtype=int)

    def update(self, new_value: float | int):
        """same as updateR but without returning any value"""
        self.updateR(new_value)

    def updateR(self, new_value: float | int):
        # same as update, also returns de discarded value
        new_data = np.asarray(new_value, dtype=self._dtype)
        self._history = self._history + np.ones(self._L, dtype=int)

        idx = np.argmax(self._history)
        drop = self._buffer[idx]
        np.concatenate(
            (self._history[:idx], self._history[idx + 1 :], [0]),
            out=self._history,
        )
        np.concatenate(
            (self._buffer[:idx], self._buffer[idx + 1 :], [0]),
            out=self._buffer,
        )

        idx = np.where(self._buffer[:-1] < new_data)[0]
        idx = self._L - 1 if not len(idx) else idx[0]
        np.concatenate(
            (self._buffer[: idx + 1], self._buffer[idx:-1]),
            out=self._buffer,
        )
        self._buffer[idx] = new_value
        np.concatenate(
            (self._history[: idx + 1], self._history[idx:-1]),
            out=self._history,
        )
        self._history[idx] = 0
        return drop

    def __str__(self):
        msg = " ".join([f"{self._history[i]:3d}" for i in range(self._L)]) + "\n"
        if self._dtype == int:
            return msg + " ".join([f"{self._buffer[i]:3d}" for i in range(self._L)])
        return msg + " ".join([f"{self._buffer[i]:5.2f}" for i in range(self._L)])


def run_filter_median(data: np.ndarray, L: int = 63) -> tuple[np.ndarray, np.ndarray]:
    assert data.ndim == 1
    buff = FilterMedian(L, dtype=data.dtype, initB=data[:L])
    N = data.shape[0]
    a = np.empty_like(data)
    for i in range(N):
        buff.update(data[i])
        a[i] = buff.getMedian()
    return a, buff.getBuffer()

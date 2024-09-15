"""
Base Buffer class
"""

import numpy as np


class Buffer:
    "Buffer"

    def __init__(self, L: int = 62, dtype: type = float, initB=None):
        if isinstance(initB, type(None)):
            self._buffer = np.zeros(L).astype(dtype)
            self._L = L
            self._dtype = dtype
        else:
            if len(np.shape(initB)) != 1:
                print(np.shape(initB))
                raise ValueError("Initial buffer should be of dimension 1")

            self._buffer = np.asarray(initB.copy(), dtype=dtype)
            self._L = len(initB)
            self._buffer.sort()
            self._dtype = dtype
        if self._L % 2 == 0:
            print(f"L={L}")
            raise ValueError("currentBuffer must have an odd number of elements")
        # Midx references the index of the median. As the buffer goes from 0 to L-1
        # the middle index is (L-1)/2
        self._Midx = int((self._L - 1) / 2)

    def __str__(self):
        print("here")
        if self._dtype == int:
            return " ".join([f"{self._buffer[i]:3d}" for i in range(self._L)])
        return " ".join([f"{self._buffer[i]:5.2f}" for i in range(self._L)])

    def getBuffer(self):
        return self._buffer.copy()

    def getMedian(self):
        return self._buffer[self._Midx]

    def getVar(self):
        return np.var(self._buffer.astype(float), ddof=1)

    def getMean(self):
        return np.mean(self._buffer.astype(float))

    def getMin(self):
        return self._buffer[0]

    def getMax(self):
        return self._buffer[-1]

    def update(self, new_value) -> None:
        raise ValueError("Not implemented")

    def updateR(self, new_value) -> int | float:
        raise ValueError("Not implemented")

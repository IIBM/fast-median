import numpy as np

from median.median import Median


def test_init():
    buff = np.array([0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 8, 9])
    print(len(buff))
    assert len(buff) % 2 == 1
    buff = Median(initB=buff)


def test_int():
    data = np.random.default_rng().integers(0, 5, size=30)
    # data = np.array([4,3,3,1,2,1,1,0])
    buff = Median(5, dtype=int)
    print(buff)
    for i in data:
        buff.update(i)
        assert (buff._buffer == np.sort(buff._buffer)).all()
        print(i, buff._buffer)


if __name__ == "__main__":
    test_init()
    test_int()

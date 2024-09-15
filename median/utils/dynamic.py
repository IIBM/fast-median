"""
Dynamic of the buffer
Shows how the buffer gets values closer to the median when time increases
"""

import time
from pathlib import Path

import numpy as np
import ray

from median.distributions import Distribution
from median.median import run_new_median_full
from median.utils.utils import pretty_time


def generate_data(
    filename: Path,
    total_steps: int,
    idxs: list[int],
    distribution: Distribution,
    iterations=100,
):
    print("Generating data")

    ray.init(num_cpus=12, num_gpus=1, include_dashboard=False)

    @ray.remote
    def runnm(data: np.ndarray, L: int, idxs: list[int]):
        N = data.shape[0] - L if len(idxs) == 0 else len(idxs)
        store = np.zeros((N, L))
        run_new_median_full(data=data, store=store, L=L, idxs=idxs)
        return store

    def run_one(L: int, iterations: int, data: np.ndarray, out: np.ndarray):
        # run one iteration
        outId = []
        for j in range(iterations):
            outId.append(runnm.remote(data=data[j], L=L, idxs=idxs))

        for j in range(iterations):
            out[j] = ray.get(outId[j])

    S = 1023
    data = np.zeros((iterations, total_steps + S))
    for j in range(iterations):
        # data es casi siempre mayor a cero, abs(data) se puede suponer gausiano
        data[j] = distribution.samples(N=total_steps + S)

    N = total_steps if len(idxs) == 0 else len(idxs)
    buff_out = np.zeros((iterations, N, S))
    start_time = time.time()
    run_one(L=S, iterations=iterations, data=data, out=buff_out)
    total_time = time.time() - start_time
    print(pretty_time(total_time))

    if np.isnan(buff_out.mean()):
        print("data is nan")
        return False

    np.savez(
        file=filename,
        total_steps=total_steps,
        idxs=idxs,
        distribution=distribution,
        buffer=buff_out,
    )
    return True


def load_data(filename: Path) -> tuple[int, list[int], Distribution, np.ndarray]:
    data_all = np.load(filename, allow_pickle=True)
    ts = data_all["total_steps"]
    idxs = data_all["idxs"]
    distr = data_all["distribution"]
    buff = data_all["buffer"]
    return (
        ts,
        idxs,
        distr,
        buff,
    )


def concatenate_data(filename: Path, tempfile: Path):
    try:
        (
            ts1,
            idxs1,
            distribution1,
            buff1,
        ) = load_data(filename=filename)
        (
            ts2,
            _,
            distribution2,
            buff2,
        ) = load_data(filename=tempfile)

        if ts1 != ts2:
            raise ValueError(f"Missmatch total steps {ts1=} {ts2=}")

        if distribution1 != distribution2:
            raise ValueError(
                f"Missmatch distributions {distribution1=} {distribution2=}"
            )

        buff = np.concatenate((buff1, buff2))

        np.savez(
            filename,
            total_steps=ts1,
            idxs=idxs1,
            distribution=distribution1,
            buffer=buff,
        )

    except Exception as e:
        print(f"ERROR {e}")

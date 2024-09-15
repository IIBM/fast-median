"""
Test run_median
"""

import time

import numpy as np

from median.filter_median import run_filter_median
from median.median import run_new_median
from median.utils.utils import run_classic_median


def test_runners():
    """multiple test"""
    N = 100000
    data = np.random.default_rng().standard_normal(N).astype(np.float32) + 10
    print(f"data type {data.dtype}")

    start = time.time()
    new_median63, _ = run_new_median(data, L=63)
    total_time = time.time() - start
    print(f"New Median L = 63, median = {new_median63[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    new_median127, _ = run_new_median(data, L=127)
    total_time = time.time() - start
    print(f"New Median L = 127, median = {new_median127[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    new_median255, _ = run_new_median(data, L=255)
    total_time = time.time() - start
    print(f"New Median L = 255, median = {new_median255[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    new_median511, _ = run_new_median(data, L=511)
    total_time = time.time() - start
    print(f"New Median L = 511, median = {new_median511[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    new_median1023, _ = run_new_median(data, L=1023)
    total_time = time.time() - start
    print(f"New Median L = 1023, median = {new_median1023[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    classic_median63, _ = run_classic_median(data, L=63)
    total_time = time.time() - start
    print(f"Classic Median L = 63, median = {classic_median63[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    classic_median127, _ = run_classic_median(data, L=127)
    total_time = time.time() - start
    print(f"Classic Median L = 127, median = {classic_median127[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    classic_median255, _ = run_classic_median(data, L=255)
    total_time = time.time() - start
    print(f"Classic Median L = 255, median = {classic_median255[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    classic_median511, _ = run_classic_median(data, L=511)
    total_time = time.time() - start
    print(f"Classic Median L = 511, median = {classic_median511[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    classic_median1023, _ = run_classic_median(data, L=1023)
    total_time = time.time() - start
    print(f"Classic Median L = 1023, median = {classic_median1023[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    filter_median63, _ = run_filter_median(data, L=63)
    total_time = time.time() - start
    print(f"Filter Median L = 63, median = {filter_median63[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    filter_median127, _ = run_filter_median(data, L=127)
    total_time = time.time() - start
    print(f"Filter Median L = 127, median = {filter_median127[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    filter_median255, _ = run_filter_median(data, L=255)
    total_time = time.time() - start
    print(f"Filter Median L = 255, median = {filter_median255[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    filter_median511, _ = run_filter_median(data, L=511)
    total_time = time.time() - start
    print(f"Filter Median L = 511, median = {filter_median511[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )

    start = time.time()
    filter_median1023, _ = run_filter_median(data, L=1023)
    total_time = time.time() - start
    print(f"Filter Median L = 1023, median = {filter_median1023[-1]}")
    print(
        f"total --- {np.floor(total_time) / 60:.0f} minutes, {total_time % 60:.2f} seconds ---"
    )


if __name__ == "__main__":
    test_runners()

import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import click
import random

import cupy as cp

from cuda_power_function import compute_power_cupy as compute_power


@click.command()
@click.option(
    "--use-range",
    is_flag=True,
    help="Flag for using the range discrepancy rather than l1",
)
@click.option(
    "--uuid",
    default=f"{int(random.random()*(10**8)):08d}",
    help="A unique integer for identifying the run",
)
@click.option("--burst-length", default=1, help="The length of the burst to run")
@click.option("--n-bursts", default=1000, help="The number of bursts to run the ")
@click.option("--threshold", default=0.5, help="The threshold for the discrep")
@click.option(
    "--show-progress",
    is_flag=True,
    help="Flag for showing the progress bar. Useful for testing",
)
@click.option("--population", default=100000, help="The population size")
def main(use_range, uuid, burst_length, n_bursts, threshold, show_progress, population):
    # fmt: off
    m_orig = np.array(
        [2284,1668,5436,5140,11109, 3640,14170,5210,3921,3679, 3473,4106,3931,9404,2403, 6637,3360,2644,1641,15860, 2740]
    )
    m_orig.sort()
    m = m_orig / np.sum(m_orig)
    # fmt: on

    if threshold == 0.67:
        threshold = 2.0 / 3

    # Note that NYS_counties does not cover all possible subsets. There is an assumption of
    # one town always voting in the affirmative. This substantially reduces the size of the
    # matrix that
    with open("NYS_counties_ontario.pkl", "rb") as f:
        A_subsets = pickle.load(f)

    subset_masks = np.zeros((len(m), len(A_subsets)), dtype=np.float64)
    for j, subset in enumerate(A_subsets):
        subset_masks[subset, j] = True

    subset_masks_bool = subset_masks.astype(np.bool_)
    subset_masks_float = subset_masks.astype(np.float64)

    mround = np.around(m * population, 0)

    in_arr = mround

    cu_m = cp.asarray(m)
    cu_u = cp.asarray(in_arr)
    cu_umat = cp.asarray(in_arr)[:, cp.newaxis]
    cu_subset_masks_bool = cp.asarray(subset_masks_bool)
    cu_subset_masks_float = cp.asarray(subset_masks_float)

    pow_fn = partial(
        compute_power,
        subset_masks_bool=cu_subset_masks_bool,
        subset_masks_float=cu_subset_masks_float,
        T=threshold,
    )

    discrep_function = lambda x: cp.sum(cp.abs(x))

    if use_range:
        discrep_function = lambda x: cp.max(x) - cp.min(x)

    curr_diff = pow_fn(u_matrix=cu_umat) - cu_m
    initial_discrep = discrep_function(curr_diff)

    best_so_far = initial_discrep
    best_u = cu_u.copy()

    file_name = f"Ontario_MCMC_discrep_L1_lenburst_{burst_length}_iters_{n_bursts}_T_{threshold}_id_{uuid}.jsonl"

    if use_range:
        file_name = f"Ontario_MCMC_discrep_range_lenburst_{burst_length}_iters_{n_bursts}_T_{threshold}_id_{uuid}.jsonl"

    cp.random.seed(seed=int(uuid))
    np.random.seed(seed=int(uuid))

    with open(f"./MCMC_results/{file_name}", "w") as f:
        print(
            f"{str('{')}\"step\": {-1}, \"discrepancy\": {initial_discrep}, \"weights\": {best_u.astype(int).tolist()}{str('}')}",
            file=f,
            flush=True,
        )

        cur_discrep = initial_discrep

        for i in tqdm(
            range(0, n_bursts * burst_length, burst_length),
            miniters=n_bursts / 1000,
            disable=not show_progress,
        ):
            # Compute the sampling probabilities
            probabilities = cp.abs(curr_diff)
            probabilities /= cp.sum(probabilities)

            # Sample indices based on the probabilities
            indices = cp.random.choice(
                len(curr_diff), size=burst_length, p=probabilities
            )

            update_values = cp.where(curr_diff[indices] < 0, 1, -1)
            values = cp.zeros_like(curr_diff, dtype=cp.int32)
            cp.add.at(values, indices, update_values)

            cp.add.at(cu_u, indices, values[indices])
            cu_umat = cu_u[:, cp.newaxis]
            new_diff = pow_fn(u_matrix=cu_umat) - cu_m
            new_discrep = discrep_function(new_diff)

            if new_discrep < best_so_far:
                best_so_far = new_discrep
                best_u = cu_u.copy()
                print(
                    f"{str('{')}\"step\": {i}, \"discrepancy\": {best_so_far}, \"weights\": {best_u.astype(int).tolist()}{str('}')}",
                    file=f,
                    flush=True,
                )

            else:
                reject_cut = np.exp(-new_discrep / cur_discrep)
                if np.random.rand() >= reject_cut:
                    cp.add.at(cu_u, indices, -values[indices])
                    continue

            cur_discrep = new_discrep
            curr_diff = new_diff


if __name__ == "__main__":
    main()

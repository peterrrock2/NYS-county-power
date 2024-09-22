"""
This is a file for containing the cuda version of the power function for use with 
NVIDIA GPUs. This can provide a 3x speedup over the numpy version.
"""

import cupy as cp


def compute_power_cupy(
    m: cp.ndarray,
    u_matrix: cp.ndarray,
    subset_masks_bool: cp.ndarray,
    subset_masks_float: cp.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Args:
        m (cupy.ndarray): An array of size (num_towns,) that contains the proportion of the
            population that each town has (this vector must sum to 1).
        u_matrix (cupy.ndarray): Array of size (1, num_towns) that contains the weights to assign
            to each town.
        subset_masks_bool (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure.
        subset_masks_float (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            the same data as the boolean masks, but as floats for ease of computation.
        T (float): The threshold for the power.

    Returns:
        cupy.ndarray:
            An array of size (num_towns,) that contains the difference between the computed voting
            power for when each town is given weight determined by the u_matrix and the proportion
            of the population that town has.
    """
    a_matrix = cp.transpose(u_matrix) @ subset_masks_float
    a_minus_T = a_matrix - T
    one_minus_a = 1 - a_matrix
    one_minus_a_minus_T = one_minus_a - T

    condition1 = cp.logical_and(u_matrix > a_minus_T, a_matrix > T)
    condition2 = cp.logical_and(u_matrix > one_minus_a_minus_T, one_minus_a > T)

    X = cp.logical_or(
        cp.logical_and(subset_masks_bool, condition1),
        cp.logical_and(~subset_masks_bool, condition2),
    )

    non_zero_count = cp.count_nonzero(X, axis=1)
    p = non_zero_count / cp.count_nonzero(X)

    return p - m

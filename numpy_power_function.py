import numpy as np


def computer_power_helper_numpy(
    m: np.ndarray,
    u_matrix: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Args:
        m (numpy.ndarray): An array of size (num_towns,) that contains the proportion of the
            population that each town has (this vector must sum to 1).
        u_matrix (numpy.ndarray): Array of size (1, num_towns) that contains the weights to assign
            to each town.
        subset_masks (numpy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure.
        T (float): The threshold for the power.

    Returns:
        numpy.ndarray:
            An array of size (num_towns,) that contains the difference between the computed voting
            power for when each town is given weight determined by the u_matrix and the proportion
            of the population that town has.
    """
    a_matrix = np.transpose(u_matrix) @ subset_masks
    a_minus_T = a_matrix - T
    one_minus_a = 1 - a_matrix
    one_minus_a_minus_T = one_minus_a - T

    condition1 = np.logical_and(u_matrix > a_minus_T, a_matrix > T)
    condition2 = np.logical_and(u_matrix > one_minus_a_minus_T, one_minus_a > T)

    X = np.logical_or(
        np.logical_and(subset_masks, condition1),
        np.logical_and(~subset_masks, condition2),
    )

    non_zero_count = np.count_nonzero(X, axis=1)
    p = non_zero_count / np.count_nonzero(X)

    return p - m

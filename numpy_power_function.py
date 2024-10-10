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
        m (numpy.ndarray): An array of size (1, num_towns) that contains the proportion of the
            population that each town has (this vector must sum to 1).
        u_matrix (numpy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks (numpy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure.
        T (float): The threshold for the power.

    Returns:
        numpy.ndarray:
            An array of size (num_towns,) that contains the difference between the computed voting
            power for when each town is given weight determined by the u_matrix and the proportion
    """
    a_matrix = np.transpose(u_matrix) @ subset_masks

    # Round all computations to 13 decimal places of precision to remove sensitivity
    # to floating point conversion and arithmetic operations.
    a_minus_T = np.around(a_matrix - T, decimals=13)
    one_minus_a = np.around(1 - a_matrix, decimals=13)
    one_minus_a_minus_T = np.around(one_minus_a - T, decimals=13)

    u_matrix_rounded = np.around(u_matrix, decimals=13)
    a_matrix = np.around(a_matrix, decimals=13)

    condition1 = np.logical_and(u_matrix_rounded > a_minus_T, a_matrix > T)
    condition2 = np.logical_and(u_matrix_rounded > one_minus_a_minus_T, one_minus_a > T)

    X = np.logical_or(
        np.logical_and(subset_masks, condition1),
        np.logical_and(~subset_masks, condition2),
    )

    non_zero_count = np.count_nonzero(X, axis=1)
    p = non_zero_count / np.count_nonzero(X)

    return p - m


def computer_power_helper_numpy_relaxed(
    m: np.ndarray,
    weights: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
):
    # This computes the power discrepancy if the functions aren’t linear threshold functions, but linear bounded functions
    # as in the Diakonikolas et al. paper.
    n = len(m)
    subset_weights = np.transpose(subset_masks).dot(weights)  # num_subsets
    subset_weights_min_T = subset_weights - T
    payoffs = np.clip(subset_weights_min_T, a_min=-1, a_max=1)  # num_subsets

    contained_matrix = 2 * subset_masks - 1  # (num_towns, num_subsets)
    banzhaf_indices = contained_matrix.dot(payoffs) / 2 ** (n - 1)

    return banzhaf_indices - m

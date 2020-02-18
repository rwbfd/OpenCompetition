"""
This module contains realization of application commands along with some minor
functions which help in training and estimation.
"""


import os

import numpy as np

from files_io import load_dataset, save_vars_dict, load_vars_dict
from adam_optimizer import AdamOptimizer
from neural_network import NeuralNetwork


# Constant paths to NN variables and normalization parameters saved on the disk.
# These paths are relative to the place in which this script is located.
NORMALIZATION_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "normalization_params.npz"
)
NET_PATH = os.path.join(os.path.dirname(__file__), "data", "net.npz")

# Training hyperparameters. These were found through trial and error.
# Training stops when that many iterations have passed.
MAX_ITERATIONS = 50000
# During the training there are attempts made to eliminate some coefficients of
# the modeled polynomial. These attempts are made after current loss falls
# below ELIMINATION_ATTEMPT_THRESHOLD value.
ELIMINATION_ATTEMPT_THRESHOLD = 1e-5
# Selected coefficient is eliminated from modeled polynomial when mean of
# absolute values of this coefficient in the current iteration is less than the
# COEFF_ELIMINATION_THRESHOLD value.
COEFF_ELIMINATION_THRESHOLD = 1e-5
# Training stops when this or smaller loss is reached.
TARGET_LOSS = 1e-25
# Multiplier applied to current polynomial degree to get size of the batch that
# will be used in learning. For example if we are currently trying to
# approximate 5th degree polynomial then we will use batches with 200 examples.
BATCH_SIZE_MULTIPLIER = 40


def normalize_mat(mat):
    """
    Normalize given matrix of values and bring all of them to mean = 0 and
    std = 1.

    Args:
        mat (np.ndarray): matrix of values that will be normalized

    Returns:
        (np.ndarray, float, float): first returned value is given matrix after
            normalization. Second returned value is mean and third is standard
            deviation.

    """
    mat_mean = np.mean(mat, axis=0)
    mat_std = np.std(mat, axis=0)
    return (mat - mat_mean) / mat_std, mat_mean, mat_std


def denormalize_coeffs(coeffs, normalization_params, polynomial_degree):
    """
    Denormalize coefficients approximated by the NN to eliminate effect of
    normalization of x and y values. Return denormalized coefficients.

    Args:
        coeffs (np.ndarray): matrix of normalized coefficients calculated by
            NN. It has to be 2d.
        normalization_params (dict): dictionary of normalization params such as
            x_mean, x_std, y_mean, y_std, x_powers_mean and x_powers_std.
        polynomial_degree (int): degree of the approximated polynomial.

    Returns:
        np.ndarray: matrix of denormalized coefficients.

    """
    # Work backwards from:
    #
    # (y - y_mean) / y_std = an * (x_n - x_n_mean) / x_n_std .... + a0
    #
    # Multiply both sides by y_std.
    coeffs = normalization_params["y_std"] * coeffs
    # Perform divisions of ak / x_k_std.
    coeffs[:, 1:] /= normalization_params["x_powers_std"][:polynomial_degree]
    # Move all free constants to a0.
    coeffs[:, 0] += (
        normalization_params["y_mean"]
        - np.sum(
            np.multiply(
                normalization_params["x_powers_mean"][:polynomial_degree],
                coeffs[:, 1:]
            )
        )
    )
    return coeffs


def try_eliminating_coeffs(normalization_params, vals, net,
                           optimizer, polynomial_degree):
    """
    Try to eliminate some coefficients from approximated polynomial.
    Coefficients are eliminated when mean of their absolute values from current
    iteration is less or equal than COEFF_ELIMINATION_THRESHOLD value. Start at
    the coefficient standing next to the biggest power of x and stop at the
    first coefficient that is not going to be eliminated. Return degree of the
    new polynomial after elimination.

    Args:
        normalization_params (dict): dictionary of normalization params such as
            x_mean, x_std, y_mean, y_std, x_powers_mean and x_powers_std.
        vals (dict): collection of values such as coefficients averages etc.
            computed during NN full forward pass.
        net (NeuralNetwork): NN object which approximates the polynomial.
        optimizer (AdamOptimizer): optimizer object which updates the NN during
            training.
        polynomial_degree (int): current degree of the modeled polynomial.

    Returns:
        int: degree of the modified polynomial. It might be the same as
            polynomial degree on input if no coefficient was eliminated.

    """
    denormalized_coeffs = denormalize_coeffs(
        vals["coeffs"], normalization_params, polynomial_degree
    )
    denormalized_coeffs_abs_avg = np.mean(np.abs(denormalized_coeffs), axis=0)

    # Try to eliminate coefficient standing next to the currently highest x
    # power in the modeled polynomial. Stop if elimination condition is not met
    # or if we try to eliminate last coefficient.
    while (
        denormalized_coeffs_abs_avg[polynomial_degree]
        < COEFF_ELIMINATION_THRESHOLD
        and polynomial_degree != 0
    ):
        net.eliminate_polynomial_degree(polynomial_degree)
        optimizer.eliminate_polynomial_degree(polynomial_degree)
        polynomial_degree -= 1
    return polynomial_degree


def shuffle_matrices(mats):
    """
    Shuffle given matrices along its first dimensions. All matrices are
    shuffled in the same way in one function call.

    Args:
        mats (list): list of np.ndarray objects - matricies - which will
            be shuffled.

    Returns:
        None

    """
    random_state = np.random.get_state()
    for mat in mats:
        np.random.set_state(random_state)
        np.random.shuffle(mat)


def train(csv_path, polynomial_degree):
    """
    Train NN to model points contained in the pointed dataset with a polynomial
    of a degree less than or equal to the given degree. Return coefficients of
    found polynomial.

    Args:
        csv_path (str): path to a .csv file which contains dataset of points.
        polynomial_degree (int): maximal degree of the modeled polynomial.

    Returns:
        list: list of coefficients of found polynomial. Most significant
            coefficient is first on this list.

    """
    x_mat, y_mat = load_dataset(csv_path)
    x_powers_mat = np.power(x_mat, range(polynomial_degree + 1))

    # Normalize only x^1 up to x^n because x^0 is always 1.
    x_powers_mat[:, 1:], x_powers_mean, x_powers_std = normalize_mat(
        x_powers_mat[:, 1:]
    )

    x_mat, x_mean, x_std = normalize_mat(x_mat)
    y_mat, y_mean, y_std = normalize_mat(y_mat)

    # Collect normalization parameters in a dict as they will be needed in
    # estimation.
    normalization_params = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean,
                            "y_std": y_std, "x_powers_mean": x_powers_mean,
                            "x_powers_std": x_powers_std}
    save_vars_dict(normalization_params, NORMALIZATION_PARAMS_PATH)

    net = NeuralNetwork(polynomial_degree)
    optimizer = AdamOptimizer(net)
    min_loss = None
    best_net_vars = {}
    best_coeffs = None
    batch_i = 0
    num_examples = x_mat.shape[0]
    for i in range(MAX_ITERATIONS):

        # Create minibatch of examples and advance batch iterator.
        batch_size = polynomial_degree * BATCH_SIZE_MULTIPLIER
        placeholders = {"x_mat": x_mat[batch_i:batch_i + batch_size],
                        "y_mat": y_mat[batch_i:batch_i + batch_size],
                        "x_powers_mat": x_powers_mat[batch_i:
                                                     batch_i + batch_size,
                                                     :polynomial_degree + 1]}
        batch_i += batch_size
        if batch_i >= num_examples:
            batch_i = 0
            shuffle_matrices([x_mat, y_mat, x_powers_mat])

        vals = net.forward_pass(placeholders)
        vals = net.loss_forward_pass(placeholders, vals)

        # Keep track of the best network approximation so far.
        if min_loss is None or vals["loss"] < min_loss:
            min_loss = vals["loss"]
            best_coeffs = vals["avg_coeffs"]
            net.backup(best_net_vars)
            if min_loss <= TARGET_LOSS:
                break

        net.backward_pass(placeholders, vals)

        # Attempt coefficients elimination after loss hits certain threshold.
        # The idea is to let the coefficients settle a bit to find out which
        # ones of them are truly unnecessary.
        if vals["loss"] < ELIMINATION_ATTEMPT_THRESHOLD:
            polynomial_degree = try_eliminating_coeffs(
                normalization_params, vals, net, optimizer, polynomial_degree
            )
            best_net_vars = {}
            net.backup(best_net_vars)

        optimizer.perform_update(net)
    save_vars_dict(best_net_vars, NET_PATH)
    coeffs = denormalize_coeffs(best_coeffs[np.newaxis, :],
                                normalization_params, polynomial_degree)
    return list(reversed(coeffs[0].tolist()))


def estimate(x):
    """
    Estimate y value for given x using previously trained NN to approximate
    this polynomial coefficients.

    Args:
        x (float): x value which will be used to calculate y with approximated
            polynomial coefficients.

    Returns:
        float: y value estimated for given x.

    """
    net_vars = load_vars_dict(NET_PATH)
    polynomial_degree = net_vars["bias_1"].shape[0] - 1
    net = NeuralNetwork.from_net_vars(net_vars)
    normalization_params = load_vars_dict(NORMALIZATION_PARAMS_PATH)
    x_powers_mat = np.power(x, range(polynomial_degree + 1))

    # Normalize input to NN as it was not trained to handle raw x values and
    # unnormalized x might skew the output coefficients.
    x_mat = np.array(
        [(x - normalization_params["x_mean"]) / normalization_params["x_std"]]
    )

    placeholders = {"x_mat": x_mat}
    vals = net.forward_pass(placeholders)
    coeffs = denormalize_coeffs(vals["coeffs"], normalization_params,
                                polynomial_degree)
    y = np.sum(np.multiply(x_powers_mat, coeffs))
    return y

"""
This module contains functions related to gradient calculation during the
back propagation.
"""


import numpy as np


def mean_grad(in_array, top_grad):
    """
    Calculate gradient of the np.mean operation with respect to each averaged
    element. Multiply it by the gradient flowing from the top according to the
    chain rule and return the result. It is assumed that the result of np.mean
    was a single element.

    Args:
        in_array (np.ndarray): numpy array which was averaged during the np.mean
            operation.
        top_grad (float): gradient of some error calculated with respect to the
            result of np.mean operation which input was in_array.

    Returns:
        np.ndarray: gradient of some error calculated with respect to each
            averaged element.

    """
    grad = top_grad * np.ones_like(in_array) / in_array.size
    return grad


def matmul_grad_of_mat_2(mat_1, top_grad):
    """
    Calculate gradient of np.matmul operation with respect to the second of the
    multiplied matrices. Multiply it by the top_grad according to the chain rule
    and return the result.

    Args:
        mat_1 (np.ndarray): first of the multiplied matrices.
        top_grad (np.ndarray): gradient of some error with respect to the result
            of the np.matmul operation.

    Returns:
        np.ndarray: gradient of some error coming from the top with respect to
            the second of the multiplied matrices.

    """
    grad = np.matmul(mat_1.T, top_grad)
    return grad


def bias_add_grad(top_grad):
    """
    Split the gradient of np.matmul(X, W) + bias for 2 parts: gradient with
    respect to the bias and gradient with respect to the np.matmul result.
    Take into account the gradient with respect to the "bias add" result which
    flows from the top.

    Args:
        top_grad: gradient of some error with respect to the result of this
            add bias operation.

    Returns:
        (np.ndarray, np.ndarray): gradient of some error with respect to the
            bias and separately with respect to the result of np.matmul.
            np.matmul gradient is first in this tuple.

    """
    bias_grad = np.sum(top_grad, axis=0)
    mat_grad = top_grad
    return mat_grad, bias_grad


def square_loss_grad(out_vals, target_vals, top_grad):
    """
    Calculate gradient of square loss operator with respect to some kind of
    a model output. Take into account gradient with respect to the result of
    this square loss operation in accordance with chain rule.

    Args:
        out_vals (np.ndarray): values output by some kind of the model.
        target_vals (np.ndarray): target values for the model.
        top_grad (np.ndarray): gradient with respect to the result of the
            square loss operation.

    Returns:
        np.ndarray: gradient of some error with respect to each element of the
            out_vals.

    """
    out_grad = np.multiply(top_grad, 2 * (out_vals - target_vals))
    return out_grad


def coeffs_variance_grad(coeffs, top_grad):
    """
    Calculate gradient of the np.var operation used on coefficients output by
    the NN. This operation was performed coefficient wise which means that the
    result of it was vector of variances of each coefficient separately.

    Args:
        coeffs: matrix of coefficients where each row is a single training
            output. This matrix was input for the variance operation.
        top_grad: gradient of some error with respect to the result of this
            operation.

    Returns:
        np.ndarray: gradient of some error with respect to each coefficient
            separately.

    """
    means = np.mean(coeffs, axis=0)
    grad = np.multiply(top_grad, 2 * (coeffs - means) / coeffs.shape[0])
    return grad


def element_wise_mul_grad(mul_mat, top_grad):
    """
    Calculate gradient of element wise np,multiply operation with respect to one
    of the multiplied matrices. Take into account gradient flowing from the top
    in accordance with the chain rule.

    Args:
        mul_mat (np.ndarray): matrix which was multiplied element wise by the
            matrix with respect to which gradient will be calculated.
        top_grad (np.ndarray): gradient of some error with respect to the
            result of this np.multiply operation.

    Returns:
        np.ndarray: gradient of some error with respect to one of the multiplied
            matrices.

    """
    grad = np.multiply(top_grad, mul_mat)
    return grad


def sum_grad(in_array, top_grad):
    """
    Calculate gradient with respect to the np.sum operation and return it
    multiplied by the gradient with respect to the result of this np.sum
    operation.

    Args:
        in_array (np.ndarray): matrix on which np.sum was used.
        top_grad (Union[np.ndarray, float]): gradient with respect to the
            result of this np.sum operation. It can be single number if the
            result of the np.sum was single number or it can be numpy array.

    Returns:
        np.ndarray: gradient of some error with respect to the in_array which
            was previously summed with np.sum.

    """
    grad = np.multiply(top_grad, np.ones_like(in_array))
    return grad

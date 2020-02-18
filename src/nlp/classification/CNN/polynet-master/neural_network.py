""" This module contains definition of the neural network class. """


import numpy as np

from gradients import (mean_grad, matmul_grad_of_mat_2, bias_add_grad,
                       square_loss_grad, coeffs_variance_grad,
                       element_wise_mul_grad, sum_grad)


class NeuralNetwork:

    def __init__(self, polynomial_degree):
        """
        Create NN variable matrices and holders for their gradients.

        Args:
            polynomial_degree: max polynomial degree with which the network
                should estimate points from the dataset.

        """
        self.vars = {
            "weights_1": np.random.normal(size=[1, polynomial_degree + 1])
                         * 0.001,
            "bias_1": np.zeros([polynomial_degree + 1, ])
        }
        self.grads = {key: np.zeros_like(self.vars[key]) for key in self.vars}

    @classmethod
    def from_net_vars(cls, net_vars):
        """
        Create NN object from existing net variables and return it.

        Args:
            net_vars (dict): dictionary of existing net variables.

        Returns:
            NeuralNetwork: NN object which contains variables loaded from the
                target .npz archive.

        """
        polynomial_degree = net_vars["bias_1"].shape[0] - 1
        net = cls(polynomial_degree)
        net.vars = net_vars
        return net

    def forward_pass(self, placeholders):
        """
        Perform forward pass through the network using given placeholders and
        return dictionary containing calculated coefficients.

        Args:
            placeholders (dict): collection of inputs to the network. It has to
                contain matrix of x values, matrix of y values and matrix of
                powers of x up to the polynomial degree which was used to create
                this network.

        Returns:
            dict: dictionary which includes only calculated matrix of
                coefficients.

        """
        vals = dict()
        vals["coeffs"] = (
            np.matmul(placeholders["x_mat"], self.vars["weights_1"])
            + self.vars["bias_1"]
        )
        return vals

    def loss_forward_pass(self, placeholders, vals):
        """
        Perform loss forward pass on top of the network output using given
        placeholders. Return dictionary containing all intermediate values such
        as coefficients average, variance, model loss etc.

        Args:
            placeholders (dict): collection of inputs to the network. It has to
                contain matrix of x values, matrix of y values and matrix of
                powers of x up to the polynomial degree which was used to create
                this network.
            vals (dict): dictionary containing coefficients calculated during
                the forward pass.

        Returns:
            dict: collection of matrices which were calculated during the loss
                forward pass. This includes estimated ys, coefficients averages
                and variances, square loss of estimated ys and sum of the mean
                square loss and mean variances of coefficients.

        """
        vals["avg_coeffs"] = np.mean(vals["coeffs"], axis=0)
        vals["coeffs_variance"] = np.var(vals["coeffs"], axis=0)
        vals["y_fit_components"] = np.multiply(placeholders["x_powers_mat"],
                                               vals["coeffs"])
        vals["y_fit"] = np.sum(vals["y_fit_components"], axis=1, keepdims=True)
        vals["square_losses"] = np.power(vals["y_fit"] - placeholders["y_mat"],
                                         2)
        vals["loss"] = np.mean(vals["square_losses"]) + np.mean(
            vals["coeffs_variance"]
        )
        return vals

    def backward_pass(self, placeholders, vals):
        """
        Perform backward pass through the network and calculate gradients of
        the main loss with respect to the NN variables.

        Args:
            placeholders (dict): collection of inputs to the network. It has to
                contain matrix of x values, matrix of y values and matrix of
                powers of x up to the polynomial degree which was used to create
                this network.
            vals (dict): collection of intermediate values calculated during the
                forward pass and loss forward pass.

        Returns:
            None

        """
        grad_of_mean_square_loss = 1
        grad_of_variance_loss = 1
        grad_of_square_losses = mean_grad(vals["square_losses"],
                                          grad_of_mean_square_loss)
        grad_of_coeffs_variance = mean_grad(vals["coeffs_variance"],
                                            grad_of_variance_loss)
        grad_of_coeffs_variance_loss = coeffs_variance_grad(
            vals["coeffs"], grad_of_coeffs_variance
        )
        grad_of_y_fit = square_loss_grad(vals["y_fit"], placeholders["y_mat"],
                                         grad_of_square_losses)
        grad_of_y_fit_components = sum_grad(vals["y_fit_components"],
                                            grad_of_y_fit)
        grad_of_coeffs_mean_square_loss = element_wise_mul_grad(
            placeholders["x_powers_mat"], grad_of_y_fit_components
        )
        grad_of_coeffs = (grad_of_coeffs_mean_square_loss
                          + grad_of_coeffs_variance_loss)
        grad_of_matmul, self.grads["bias_1"] = bias_add_grad(grad_of_coeffs)
        self.grads["weights_1"] = matmul_grad_of_mat_2(placeholders["x_mat"],
                                                       grad_of_matmul)

    def backup(self, backup_net_vars):
        """
        Copy current net variables to external dictionary for saving purposes.
        If the dictionary is empty create new copies in it. Else copy net
        variables to an already existing places in memory.

        Args:
            backup_net_vars (dict):

        Returns:
            None

        """
        for key in self.vars:
            if key not in backup_net_vars:
                backup_net_vars[key] = np.empty_like(self.vars[key])
            backup_net_vars[key][:] = self.vars[key][:]
        return backup_net_vars

    def eliminate_polynomial_degree(self, eliminated_degree):
        """
        Delete network parameters responsible for calculation of given
        polynomial degree. It will keep this parameter at 0 in all next
        iterations given that correct column of x_powers_mat will be cleared
        too.

        Args:
            eliminated_degree (int): degree of the approximated polynomial
                which will be cleared. For example when this value is 3 then
                coefficient standing next to the x^3 will be deleted.

        Returns:
            None

        """
        mask = np.ones_like(self.vars["bias_1"], dtype=bool)
        mask[eliminated_degree] = False
        self.vars["weights_1"] = self.vars["weights_1"][:, mask]
        self.vars["bias_1"] = self.vars["bias_1"][mask]
        self.grads["weights_1"] = self.grads["weights_1"][:, mask]
        self.grads["bias_1"] = self.grads["bias_1"][mask]

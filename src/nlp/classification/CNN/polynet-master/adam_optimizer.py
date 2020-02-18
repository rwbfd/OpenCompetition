"""
This module contains class that represents Adam optimizer.

For more info on Adam optimizer go to: https://arxiv.org/abs/1412.6980

"""


import numpy as np


class AdamOptimizer:

    def __init__(self, net, alfa=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer variables, collect them in one dictionary and
        return it.

        Args:
            net (NeuralNetwork):
            alfa (float): learning rate parameter
            beta_1 (float): diminishing multiplier of accumulated average of
                gradients
            beta_2 (float): diminishing multiplier of accumulated variance of
                gradients
            epsilon (float): small number necessary to avoid division by 0 in
                the update

        Returns:
            None

        """
        self.timestep = 0
        self.alfa = alfa
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.grads_avgs = {key: np.zeros_like(net.vars[key])
                           for key in net.vars}
        self.grads_variances = {key: np.zeros_like(net.vars[key])
                                for key in net.vars}

    def perform_update(self, net):
        """
        Update NN variables with one Adam optimizer step.

        Args:
            net (NeuralNetwork): NN object which will be updated by this
                optimizer.

        Returns:
            None

        """
        self.timestep += 1
        for key in net.vars:
            self.grads_avgs[key] = (self.beta_1 * self.grads_avgs[key]
                                    + (1 - self.beta_1) * net.grads[key])
            self.grads_variances[key] = (
                self.beta_2 * self.grads_variances[key]
                + (1 - self.beta_2) * np.power(net.grads[key], 2)
            )
            alfa_t = (self.alfa * np.sqrt(1 - self.beta_2 ** self.timestep)
                      / (1 - self.beta_1 ** self.timestep))
            net.vars[key] -= (
                alfa_t * self.grads_avgs[key]
                / (np.sqrt(self.grads_variances[key]) + self.epsilon)
            )

    def eliminate_polynomial_degree(self, eliminated_degree):
        """
        Delete optimizer parameters associated with NN parameters responsible
        for calculation of given polynomial degree.

        Args:
            eliminated_degree (int): degree of the approximated polynomial
                which will be cleared. For example when this value is 3 then
                optimizer variables related to NN variables which take part in
                calculation of coefficient standing next to the x^3 will be
                deleted.

        Returns:
            None

        """
        mask = np.ones_like(self.grads_avgs["bias_1"], dtype=bool)
        mask[eliminated_degree] = False
        self.grads_avgs["weights_1"] = self.grads_avgs["weights_1"][:, mask]
        self.grads_avgs["bias_1"] = self.grads_avgs["bias_1"][mask]
        self.grads_variances["weights_1"] = (self.grads_variances["weights_1"]
                                             [:, mask])
        self.grads_variances["bias_1"] = self.grads_variances["bias_1"][mask]

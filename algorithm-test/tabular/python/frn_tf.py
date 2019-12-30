import tensorflow as tf
import tensorflow.keras as k


class FRN(k.layers.Layer):
    """
        class to be used as a tf.keras.layers.Layer for a tf.keras.models.Sequential / tf.keras.models.Model model

        implements a filter response normalization layer (as described in https://arxiv.org/abs/1911.09737)
        used in convolutional neural networks (input shape should be 4 dimensional)

        ...

        Attributes
        ----------
        eps : float
            small number to prevent numerical instability

        Methods
        -------
        build
            called when model.build is called
        call
            used in forward pass, computes output of layer
    """
    def __init__(self, eps=1e-6):
        """
            initializes the layer hyper parameters
            called when layer is added to model

            Variables
            ----------
            eps : float
                small number to prevent numerical instability
        """
        super(FRN, self).__init__()
        self.eps = eps  # parameter to prevent numerical instability (division by 0)

    def build(self, input_shape):
        """
            initializes layers trainable parameters when model is built
        """
        # all parameters of shape (1, 1, 1, #filters)
        self.beta = self.add_variable('beta', shape=(1, 1, 1, input_shape[3]))
        self.gamma = self.add_variable('gamma', shape=(1, 1, 1, input_shape[3]))
        self.tau = self.add_variable('tau', shape=(1, 1, 1, input_shape[3]))

    def call(self, x):
        """
            computes output of layer in forward pass of model
        """
        # transform inputs
        v2 = tf.math.reduce_mean(x, (1, 2), True)  # square mean of each filter
        xhat = x/tf.math.sqrt(v2 + self.eps)  # divide input filter by respective means
        y = self.gamma * xhat + self.beta  # scale and shift the filter activations
        z = tf.math.maximum(y, self.tau)  # apply ReLU like activation (with shifted minimum)
        return z

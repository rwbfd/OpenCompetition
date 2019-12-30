import tensorflow as tf
import tensorflow_addons as tfa

class Linear(tf.keras.Model):
    '''tf linear layer'''
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output




class NormedLinear(tf.keras.Model):
    def __init__(self,input_dim,out_dim,momentum=0.1):
        super(NormedLinear,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.momentum = momentum
        self._build_model()

    # def _build_model(self):
    #     self.linear = nn.utils.weight_norm(Linear(self.input_dim,self.out_dim))
    #     #self.bias = nn.Parameter(nn.Tensor)
    #     self.bias = tf.Variable(tf.random_normal([self.out_dim]))
    #     self.register_buffer('running_mean',tf.zeros(self.out_dim))
    #     self.reset_parameter()

    def _build_model(self):
        #self.linear = tfa.layers.WeightNormalization(tf.keras.Sequential(self.input_dim, self.out_dim))
        self.linear = tfa.layers.WeightNormalization(Linear(self.input_dim, self.out_dim))

        self.bias = self.add_variable('bias',shape=[self.input_dim[-1]])
        self.register_buffer('running_mean', tf.zeros(self.out_dim))
        self.reset_parameter()


    def reset_parameter(self):
        self.running_mean.zero_()
        self.bias.data.zero_()

    def forward(self,inputs):
        inputs = self.linear(inputs)

        if self.training:
            avg = tf.mean(inputs,dim=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * avg.data
        else:
            avg = tf.Variable(self.running_mean,requires_grad=False)
        out = inputs - avg + self.bias

        return out
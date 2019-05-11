import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression

class HiddenLayer(object):
    def __init__(self, inpt, n_in, n_out, W=None, b=None,activation=tf.nn.sigmoid):
        if W is None:
            bound_val = 4.0*np.sqrt(6.0/(n_in + n_out))
            W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-bound_val, maxval=bound_val),dtype=tf.float32, name="W")
        if b is None:
            b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32, name="b")

        self.W = W
        self.b = b
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, inpt, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(inpt, n_in=n_in, n_out=n_hidden)
        self.outputLayer = LogisticRegression(self.hiddenLayer.output, n_in=n_hidden,n_out=n_out)
        # L1 norm
        self.L1 = tf.reduce_sum(tf.abs(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.abs(self.outputLayer.W))
        # L2 norm
        self.L2 = tf.reduce_sum(tf.square(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.square(self.outputLayer.W))
        # cross_entropy
        self.cost = self.outputLayer.cost
        self.accuracy = self.outputLayer.accuarcy
        self.params = self.hiddenLayer.params + self.outputLayer.params
        self.input = inpt
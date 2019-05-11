"""
Logistic Regression
"""
import numpy as np
import tensorflow as tf
import input_data

class LogisticRegression(object):
    def __init__(self, inpt, n_in, n_out):
        self.W = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32)
        self.output = tf.nn.softmax(tf.matmul(inpt, self.W) + self.b)
        self.y_pred = tf.argmax(self.output, axis=1)
        self.params = [self.W, self.b]

    #损失函数
    def cost(self, y):
        # cross_entropy
        return -tf.reduce_mean(tf.reduce_sum(y * tf.log(self.output), axis=1))

    def accuarcy(self, y):
        correct_pred = tf.equal(self.y_pred, tf.argmax(y, axis=1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
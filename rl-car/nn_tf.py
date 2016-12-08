import tensorflow as tf
import numpy as np


class NN():

    def __init__(self, input_shape, output_shape):
        """In case of Q learning, the input_shape is size of state space and the
        output_shape denotes the size of the action space. """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def placeholder_inputs(self, batch_size):
        """ self.input_shape inputs and 3 outputs. But the label for each state
        would be the correct action's index. """

        input_placeholder = tf.placeholder(
            tf.float32, shape=(batch_size, self.input_shape))
        nextQ = tf.placeholder(tf.float32, shape=(
            batch_size, self.output_shape))
        return input_placeholder, nextQ

    def inference(self, input_placeholder, h1_size, h2_size):
        """ Layout the shape of the NN in the inference. """

        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal(
                [self.input_shape, h1_size], stddev=1.0 / self.input_shape))
            hidden1 = tf.nn.relu(tf.matmul(input_placeholder, weights))

        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(tf.truncated_normal(
                [h1_size, h2_size], stddev=1.0 / self.input_shape))
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

        # Linear
        with tf.name_scope('linear'):
            weights = tf.Variable(tf.truncated_normal(
                [h2_size, self.output_shape], stddev=1.0 / self.input_shape))
            Qout = tf.matmul(hidden2, weights)
            #predict = tf.argmax(Qout, 1)
        return Qout

    def loss_val(self, Qout, nextQ):
        """ Qout and nextQ both are the Q values for the same action just
        calculated in different ways, that's all."""
        loss = tf.reduce_mean(tf.pow(tf.sub(nextQ, Qout), 2.0))
        return loss

    def training(self, loss, learning_rate):
        """ Returns the ops needed to minimize the loss via Gradient Descent. """
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op

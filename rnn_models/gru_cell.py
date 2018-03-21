#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self implemented GRUcell
Initialize weights with Xavier
Code adapted from CS224N
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("GRUCell")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """
    GRU cell class for Tensorflow

    Initialize weights with Xavier's initialization
    Initialize biases to 0
    Initial hidden state to 0
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """
        Updates the state using the previous @state and @inputs.

        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            # Variable intialization
            ## Reset gate
            W_r = tf.get_variable(name = "W_r", shape = [self.input_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            U_r = tf.get_variable(name = "U_r", shape = [self.state_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            b_r = tf.get_variable(name = "b_r", shape = [self.state_size, ], initializer = tf.constant_initializer(0))

            ## Update gate
            W_z = tf.get_variable(name = "W_z", shape = [self.input_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            U_z = tf.get_variable(name = "U_z", shape = [self.state_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable(name = "b_z", shape = [self.state_size, ], initializer = tf.constant_initializer(0))

            ## Alternate new state
            W_o = tf.get_variable(name = "W_o", shape = [self.input_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable(name = "U_o", shape = [self.state_size, self.state_size], initializer = tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable(name = "b_o", shape = [self.state_size, ], initializer = tf.constant_initializer(0))

            # Forward prop
            ## Update gate
            z_t = tf.add(tf.add(tf.matmul(inputs, W_z), tf.matmul(state, U_z)), b_z)
            z_t = tf.sigmoid(z_t)
            ## Reset gate
            r_t = tf.add(tf.add(tf.matmul(inputs, W_r), tf.matmul(state, U_r)), b_r)
            r_t = tf.sigmoid(r_t)
            ## Alternate new state
            o_t = tf.add(tf.add(tf.matmul(inputs, W_o), tf.matmul(tf.multiply(r_t, state), U_o)), b_o)
            o_t = tf.tanh(o_t)
            ## New state
            new_state = tf.add(tf.multiply(z_t, state), tf.multiply(tf.subtract(1., z_t), o_t))

        output = new_state
        return output, new_state

def test_gru_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_gru_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None,3))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,2))

            with tf.variable_scope("gru"):
                tf.get_variable("W_r", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_r", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_r",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("W_z", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_z", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_z",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("W_o", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_o", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_o",  initializer=np.array(np.ones(2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            cell = GRUCell(3, 2)
            y_var, ht_var = cell(x_placeholder, h_placeholder, scope="gru")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                x = np.array([
                    [0.4, 0.5, 0.6],
                    [0.3, -0.2, -0.1]], dtype=np.float32)
                h = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                y = np.array([
                    [ 0.320, 0.555],
                    [-0.006, 0.020]], dtype=np.float32)
                ht = y

                y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))

                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

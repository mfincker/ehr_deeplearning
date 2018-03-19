#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
RNN models for EHR project
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
import os
from datetime import datetime
import cPickle as pickle
from random import random

import tensorflow as tf
import numpy as np

from rnn_model import DLModel


logger = logging.getLogger("RNN")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, args):
        self.n_features = 1 # 1 code at a time is fed
        self.max_length = 60 # longest sequence to parse
        # self.n_classes = 2
        self.dropout = 0.5 # not used currently
        self.embed_type = "one-hot"
        self.embed_size = 10000
        self.hidden_size = 10
        self.batch_size = 64
        self.n_epochs = 5
        self.max_grad_norm = 10.
        self.lr = 0.001
        self.clip_gradients = False
        self.cell = args.cell # rnn / gru / lstm
        self.embed_type = args.embed_type
        self.embed_size = args.embed_size
        self.clip_gradients = args.clip_gradients
        self.pos_weight = 1

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.model_output = self.output_path + "model.weights"
        self.loss_output = self.output_path + "losses.tsv"
        self.log_output = self.output_path + "log"

def pad_sequences(data, max_length):
    """Ensures each input seqeunce in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    For every code sequence in @data,
    (a) create a new sequence which appends zeros  until
    the sequence is of length @max_length. If the sequence is longer
    than @max_length, simply truncate the sequence to be @max_length
    long from the end.
    (b) create a _hidden_masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.
    (c) create a scalar representing the index of the last code in the originial code sequence

    Example: for the sequence : [4, 6, 7], and max_length = 5, we would construct
        - a new sentence: [4, 6, 7, 0, 0]
        - a masking seqeunce: [True, True, True, False, False]
        - a index: 2

    Args:
        data: is a list of code sequences . @code sequence is a list
            containing the code indeces in the sequence. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (code sequence', mask, index).
        Each of code sequence' and mask are of length @max_length. Index is a scalar.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = 0 # correspond to UKN code

    for sequence in data:
        ### YOUR CODE HERE (~4-6 lines)
        n = len(sequence)
        if n >= max_length:
            ret.append((sequence[n - max_length: n], max_length - 1))

        else:
            sequence = sequence + [zero_vector] * (max_length - n)
            ret.append((sequence, n-1))

        ### END YOUR CODE ###
    return ret

class RNNModel(DLModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, 1), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32
        """

        self.input_placeholder = tf.placeholder(tf.int32, shape = (None, self.config.max_length), name = "input_placeholder")
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (None), name = "labels_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, name = "dropout_placeholder")
        self.index_placeholder = tf.placeholder(tf.int32, shape = (None), name = "index_placeholder")

    def create_feed_dict(self, inputs_batch, index_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {   self.input_placeholder: inputs_batch,
                        self.dropout_placeholder: dropout,
                        self.index_placeholder: index_batch }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        - if embed_type = "one-hot": embed the codes as one-hot vectors
        - if embed_type = "pretrained": embed the codes as trainable pretrained embeding vectors

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """

        if self.config.embed_type == "one-hot":
            embeddings = tf.cast(tf.one_hot(
                                    self.input_placeholder,
                                    self.config.embed_size,
                                    on_value=1,
                                    off_value=0,
                                    axis=-1,
                                    # dtype=tf.int32,
                                    name="one_hot_embedded_input"
                                ), tf.float32)
        else:
            embeddings = tf.Variable(self.pretrained_embeddings, dtype = tf.float32, name = "vocabulary", trainable=True)
            embeddings = tf.nn.embedding_lookup(params = embeddings, ids = self.input_placeholder)
            embeddings = tf.reshape(tensor = embeddings, shape = [-1, self.max_length, self.config.n_features * self.config.embed_size])                                                  
                   
        return embeddings

    def add_prediction_op(self):
        """

        Returns:
            pred: tf.Tensor of shape (batch_size, 1)
        """

        preds = [] # Predicted output at each timestep should go here!

        if self.config.cell == "rnn":
            cell = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        else:
            raise ValueError("Unsupported cell type.")

        x = self.add_embedding()

        preds, _ = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32) # size (batch_size, time_step, hidden_size)

        # Only keep last state that corresponds to a real code - using index_placeholder
        batch_size = tf.shape(preds)[0]
        max_length = tf.shape(preds)[1]
        state_size = int(preds.get_shape()[2])
        idx = tf.range(0, batch_size) * max_length + (self.index_placeholder)
        preds = tf.reshape(preds, [-1, state_size]) # reshape to (batch_size * time_step, hidden_size)
        preds = tf.gather(preds, idx) # shape (batch_size, hidden_size)

        # Prediction layer
        U = tf.get_variable(name = "U", shape = [self.config.hidden_size, 1], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = "b", shape = [1])

        preds = tf.add(tf.matmul(preds, U), b)

        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, 1) containing the output of the neural
                  network before the sigmoid.

        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.nn.weighted_cross_entropy_with_logits(targets = tf.cast(self.labels_placeholder, tf.float32),
                                                        logits = preds,
                                                        pos_weight = self.config.pos_weight)
        loss = tf.reduce_mean(loss) 
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. 
        Also implement gradient clipping.


        Args:
            loss: Loss tensor, from sigmoid_loss.
        Returns:
            train_op: The Op for training.
        """

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)

        gradients = optimizer.compute_gradients(loss)
        gradients, variables = zip(*gradients)

        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        
        # self.grad_norm = tf.global_norm(gradients)
        gradients = zip(gradients, variables)            
        
        train_op = optimizer.apply_gradients(gradients)

        return train_op

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.config.max_length)

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def predict_on_batch(self, sess, inputs_batch, index_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, index_batch=index_batch)
        predictions = sess.run(tf.cast(tf.reshape(self.pred, [-1]) > 0, tf.int32 ), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, index_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, index_batch=index_batch,
                                     dropout=self.config.dropout)
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
        return loss

    def __init__(self, config, pretrained_embeddings = None, report=None):
        super(RNNModel, self).__init__(config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.index_placeholder = None
        self.dropout_placeholder = None

        self.build()

def test_pad_sequences():
    Config.n_features = 2
    data = [
            [4, 1, 6, 8, 7, 9],
            [3, 0, 3],
        ]
    ret = [
        ([1, 6, 8, 7, 9], 4),
        ([3, 0, 3, 0, 0], 2)
        ]

    ret_ = pad_sequences(data, 5)
    assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    for i in range(2):
        assert len(ret_[i]) == 2, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
        for j in range(2):
            assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)

def do_test1(_):
    logger.info("Testing pad_sequences")
    test_pad_sequences()
    logger.info("Passed!")

def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args) # TODO double check which params are set / need to be set
    train_x, train_y, dev_x, dev_y, test_x, test_y, max_length = load_and_preprocess_data(args)
    config.max_length = min(max_length, config.max_length)

    if config.embed_type != "one-hot":
        embeddings = load_embeddings(args, helper)
        config.embed_size = embeddings.shape[1]
    else:
        embeddings = None
        config.embed_size = args.embed_size

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train_x, train_y, dev_x, dev_y)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def load_and_preprocess_data(args):

    def remove_ukn_zero_most_common(xStream, yStream, idxToRemoveStream):
        x = pickle.load(xStream)
        y = pickle.load(yStream)
        idx_to_remove = pickle.load(idxToRemoveStream)
        idx_to_remove.add(0)

        x_ = [[c for c in seq if (c not in idx_to_remove)] for seq in x]
        zero_code = [i for i, c in enumerate(x_) if len(c) == 0]
        x_ = [c for i, c in enumerate(x_) if i not in zero_code]
        y_ = [l for i, l in enumerate(y) if i not in zero_code]

        # Remove 80% of the negative examples
        x_sampled = []
        y_sampled = []
        for c, l in zip(x_, y_):
            if l == 0 and random() < 0.8:
                continue
            else:
                x_sampled.append(c)
                y_sampled.append(l)

        max_length = max([len(v) for v in x_sampled])
        assert len(x_sampled) == len(y_sampled), "train x and y don't have the same length."

        return x_sampled, y_sampled, max_length

    logger.info("Loading training data...")
    train_x, train_y, max_length_train = remove_ukn_zero(args.train_x, args.train_y)
    logger.info("Done. Read %d sentences", len(train_y))

    logger.info("Loading dev data...")
    dev_x, dev_y, max_length_dev = remove_ukn_zero(args.dev_x, args.dev_y)
    logger.info("Done. Read %d sentences", len(dev_y))

    logger.info("Loading test data...")
    test_x, test_y, max_length_test = remove_ukn_zero(args.test_x, args.test_y)
    logger.info("Done. Read %d sentences", len(test_y))

    max_length = max(max_length_train, max_length_dev, max_length_test)
    
    return train_x, train_y, dev_x, dev_y, test_x, test_y, max_length

def load_embeddings(args):
    raise NotImplementedError("Implement when I get embeddings from Sara")

def do_train(args):
    # Set up some parameters.
    config = Config(args)
    train_x, train_y, dev_x, dev_y, test_x, test_y, max_length = load_and_preprocess_data(args)
    config.max_length = min(max_length, config.max_length)

    if config.embed_type != "one-hot":
        embeddings = load_embeddings(args, helper)
        config.embed_size = embeddings.shape[1]
    else:
        embeddings = None
        config.embed_size = args.embed_size

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)
        logger.info("params:")
        logger.info('\n'.join("%s: %s" % item for item in vars(model.config).items()))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            _, losses = model.fit(session, saver, train_x, train_y, dev_x, dev_y)

            if report:
                report.log_output(model.output(session, dev_x, dev_y))
                report.save()
            else:
                # Save losses in a text file.
                with open(model.config.loss_output, 'w') as f:
                    for i, batch_loss in enumerate(losses):
                        for l in batch_loss:
                            f.write(str(i) + "\t" + str(l) + "\n")

# def do_evaluate(args):
#     config = Config(args)
#     helper = ModelHelper.load(args.model_path)
#     input_data = read_conll(args.data)
#     embeddings = load_embeddings(args, helper)
#     config.embed_size = embeddings.shape[1]

#     with tf.Graph().as_default():
#         logger.info("Building model...",)
#         start = time.time()
#         model = RNNModel(helper, config, embeddings)

#         logger.info("took %.2f seconds", time.time() - start)

#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver()

#         with tf.Session() as session:
#             session.run(init)
#             saver.restore(session, model.config.model_output)
#             for sentence, labels, predictions in model.output(session, input_data):
#                 predictions = [LBLS[l] for l in predictions]
#                 print_sentence(args.output, sentence, labels, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_test1)

    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Training data x")
    command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Training data y")
    command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Dev data x")
    command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Dev data y")
    command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Test data x")
    command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Test data y")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-et', '--embed_type', choices=["one-hot", "embed"], default="one-hot", help="type of embeddings")
    command_parser.add_argument('-es', "--embed_size", type = int, default=10000, help="Size of embeddings")
    command_parser.add_argument('-cg', "--clip_gradients", type = bool, default=False, help="Enable gradient clipping")
    command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    # command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Training data x")
    # command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Training data y")
    # command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Dev data x")
    # command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Dev data y")
    # command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Test data x")
    # command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Test data y")
    command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_x.pyc", help="Training data x")
    command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_y.pyc", help="Training data y")
    command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_x.pyc", help="Dev data x")
    command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_y.pyc", help="Dev data y")
    command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_x.pyc", help="Test data x")
    command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_y.pyc", help="Test data y")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-et', '--embed_type', choices=["one-hot", "embed"], default="one-hot", help="type of embeddings")
    command_parser.add_argument('-es', "--embed_size", type = int, default=10000, help="Size of embeddings")
    command_parser.add_argument('-cg', "--clip_gradients", type = bool, default=False, help="Enable gradient clipping")
    command_parser.add_argument('-idx', "--idx_to_remove", type = argparse.FileType('rb'), default="../dataset/idx_most_common_40.pyc", help="list of indices to remove from the dataset")
    command_parser.set_defaults(func=do_train)

    # command_parser = subparsers.add_parser('evaluate', help='')
    # command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Training data")
    # command_parser.add_argument('-m', '--model-path', help="Training data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    # command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    # command_parser.set_defaults(func=do_evaluate)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

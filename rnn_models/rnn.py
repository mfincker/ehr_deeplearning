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
from random import random, seed, randint

import tensorflow as tf
import numpy as np

from rnn_model import RNNModel


logger = logging.getLogger("RNN")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

#################
# Config object #
#################

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, args):
        self.n_features = 1 # 1 code at a time is fed
        self.max_length = args.max_length # longest sequence to parse
        self.embed_type = "one-hot"
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.batch_size = 16
        self.n_epochs = 50
        self.max_grad_norm = 10.
        self.lr = args.learning_rate
        self.clip_gradients = False
        self.cell = args.cell # rnn / gru / lstm
        self.embed_type = args.embed_type
        self.embed_size = args.embed_size
        self.clip_gradients = args.clip_gradients
        self.pos_weight = args.pos_weight
        self.idx_to_remove = args.idx_to_remove
        self.feature_size = args.feature_size
        self.train_x = args.train_x.name
        self.adaptive_lr = args.adaptive_lr

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}_{}/".format(self.cell, datetime.now(), str(randint(0, 9)))

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.model_output = self.output_path + "model.weights"
        self.loss_output = self.output_path + "losses.tsv"
        self.log_output = self.output_path + "log"

####################
# Helper functions #
####################

def load_and_preprocess_data(args, train = True, dev = True, test = True):
    '''
    Load and preprocess the data from files to meeth the RNN requirement.
    The data is filtered to remove the most common codes (@args.idx_to_remove)
    and in case of train set, sampled down to only keep 20% of the negative 
    examples (to make training faster / easier)

    Args:
        @args: object - command parser arguments. Can contain:
                - args.train_x: file handler for pickeld train data
                - args.train_y:file handler for pickeld train label
                - args.dev_x: file handler for pickeld dev data
                - args.dev_y:file handler for pickeld dev label
                - args.test_x: file handler for pickeld test data
                - args.test_y:file handler for pickeld test label
                - args.idx_to_remove: largest index of the most common code to remove from the dataset
        @train: bool - if True, expects train data/labels and process them
        @dev: bool - if True, expects dev data/labels and process them
        @test: bool - if True, expects test data/labels and process them

    Returns:
        Train and/or Dev and/or Test datasets, cleaned and sampled down. 
        Also returns the lenght of the longest code sequence in the dataset.
    '''
    def remove_ukn_zero_most_common(xStream, yStream, idx_to_remove, trainSet):
        x = pickle.load(xStream)
        y = pickle.load(yStream)

        x_ = [[c for c in seq if (c > idx_to_remove)] for seq in x]
        zero_code = [i for i, c in enumerate(x_) if len(c) == 0]
        x_ = [c for i, c in enumerate(x_) if i not in zero_code]
        y_ = [l for i, l in enumerate(y) if i not in zero_code]

        x_sampled = []
        y_sampled = []
        for c, l in zip(x_, y_):
            if trainSet: # Remove 80% of the examples
                if random() < 0.8:
                    continue
                else:
                    x_sampled.append(c)
                    y_sampled.append(l)
            else: # Keep everything
                x_sampled = x_
                y_sampled = y_


        max_length = max([len(v) for v in x_sampled])
        assert len(x_sampled) == len(y_sampled), "train x and y don't have the same length."

        return x_sampled, y_sampled, max_length

    lengths = []
    train_x, train_y, dev_x, dev_y, test_x, test_y = [None] * 6

    if train:
        logger.info("Loading training data...")
        train_x, train_y, max_length_train = remove_ukn_zero_most_common(args.train_x, args.train_y, args.idx_to_remove, True)
        logger.info("Done. Read %d sentences", len(train_y))
        lengths.append(max_length_train)

    if dev:
        logger.info("Loading dev data...")
        dev_x, dev_y, max_length_dev = remove_ukn_zero_most_common(args.dev_x, args.dev_y, args.idx_to_remove, False)
        logger.info("Done. Read %d sentences", len(dev_y))
        lengths.append(max_length_dev)

    if test:
        logger.info("Loading test data...")
        test_x, test_y, max_length_test = remove_ukn_zero_most_common(args.test_x, args.test_y, args.idx_to_remove, False)
        logger.info("Done. Read %d sentences", len(test_y))
        lengths.append(max_length_test)

    max_length = max(lengths)
    
    return train_x, train_y, dev_x, dev_y, test_x, test_y, max_length

def load_embeddings(args):
    raise NotImplementedError("Implement when I get embeddings")

##########################
# Command line functions #
##########################

def do_test_model_build(args):
    '''
    Load simple data and build an RNN model on it.
    Use to debug RNNModel.
    '''
    logger.info("Testing implementation of RNNModel")
    config = Config(args) # TODO double check which params are set / need to be set
    train_x, train_y, dev_x, dev_y, _, _, max_length = load_and_preprocess_data(args, test = False)
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


def do_train(args):
    # Set up some parameters.
    config = Config(args)
    train_x, train_y, dev_x, dev_y, _, _, max_length = load_and_preprocess_data(args, test = False)
    
    # Downsample the dataset - too long to train on full data
    keep_train = 5000
    keep_dev = 500
    logger.info("keeping only %d train examples and %d dev examples" % (keep_train, keep_dev))
    train_x = train_x[:keep_train]
    train_y = train_y[:keep_train]
    dev_x = dev_x[:keep_dev]
    dev_y = dev_y[:keep_dev]

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

def do_evaluate(args):
    seed(41)
    config = Config(args)
    _, _, _, _, test_x, test_y, max_length = load_and_preprocess_data(args, train = False, dev = False)
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
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            entity_scores = model.evaluate(session, test_x, test_y)
            logger.info("Accuracy/Precision/Recall/F1: %.2f/%.2f/%.2f/%.2f", *entity_scores)
 
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test_build', help='')
    command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Training data x")
    command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Training data y")
    command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Dev data x")
    command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Dev data y")
    command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Test data x")
    command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Test data y")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file") # add when I have embeddings
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-et', '--embed_type', choices=["one-hot", "embed"], default="one-hot", help="type of embeddings")
    command_parser.add_argument('-es', "--embed_size", type = int, default=10000, help="Size of embeddings")
    command_parser.add_argument('-cg', "--clip_gradients", type = bool, default=False, help="Enable gradient clipping")
    command_parser.add_argument('--max_length', type = int, default = 50, help = "Max length of code sequence to consider")
    command_parser.add_argument('--hidden_size', type = int, default = 100, help = "RNNcell hidden state size")
    command_parser.add_argument('-lr', "--learning_rate", type = float, default = 0.001, help = "Learning rate value")
    command_parser.add_argument('--pos_weight', type = float, default = 1, help="Weight for positive examples in the loss")
    command_parser.add_argument('-idx', "--idx_to_remove", type = int, default=40, help="Index of the last most common code to remove")
    command_parser.add_argument('-fs', "--feature_size", type = int, default=500, help = "Size of feature space")
    command_parser.add_argument('-alr', "--adaptive_lr", type = bool, default = False, help = "Enable adaptive lr")
    command_parser.set_defaults(func=do_test_model_build)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.train_x.pyc", help="Training data x")
    command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.train_y.pyc", help="Training data y")
    command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_x.pyc", help="Dev data x")
    command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.dev_y.pyc", help="Dev data y")
    command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.test_x.pyc", help="Test data x")
    command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180daysdev_x.test_y.pyc", help="Test data y")
    # command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_x.pyc", help="Training data x")
    # command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_y.pyc", help="Training data y")
    # command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_x.pyc", help="Dev data x")
    # command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_y.pyc", help="Dev data y")
    # command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_x.pyc", help="Test data x")
    # command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_y.pyc", help="Test data y")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "gru_tf", "lstm", "multi_gru", "multi_rnn"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-et', '--embed_type', choices=["one-hot", "embed"], default="one-hot", help="type of embeddings")
    command_parser.add_argument('-es', "--embed_size", type = int, default=10000, help="Size of embeddings")
    command_parser.add_argument('-fs', "--feature_size", type = int, default=500, help = "Size of feature space")
    command_parser.add_argument('-cg', "--clip_gradients", type = bool, default=False, help="Enable gradient clipping")
    command_parser.add_argument('-idx', "--idx_to_remove", type = int, default=40, help="Index of the last most common code to remove")
    command_parser.add_argument('-lr', "--learning_rate", type = float, default = 0.001, help = "Learning rate value")
    command_parser.add_argument('-alr', "--adaptive_lr", type = bool, default = False, help = "Enable adaptive lr")
    command_parser.add_argument('--hidden_size', type = int, default = 100, help = "RNNcell hidden state size")
    command_parser.add_argument('--max_length', type = int, default = 50, help = "Max length of code sequence to consider")
    command_parser.add_argument('--pos_weight', type = float, default = 1, help="Weight for positive examples in the loss")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-tx','--train_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_x.pyc", help="Training data x")
    command_parser.add_argument('-ty','--train_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.train_y.pyc", help="Training data y")
    command_parser.add_argument('-dx','--dev_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_x.pyc", help="Dev data x")
    command_parser.add_argument('-dy','--dev_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.dev_y.pyc", help="Dev data y")
    command_parser.add_argument('-sx','--test_x', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_x.pyc", help="Test data x")
    command_parser.add_argument('-sy','--test_y', type=argparse.FileType('r'), default="../dataset/full_data10000_indexes_180days.test_y.pyc", help="Test data y")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('model_path', help="Training data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "gru_tf", "lstm", "multi_gru", "multi_rnn"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-et', '--embed_type', choices=["one-hot", "embed"], default="one-hot", help="type of embeddings")
    command_parser.add_argument('-es', "--embed_size", type = int, default=10000, help="Size of embeddings")
    command_parser.add_argument('-fs', "--feature_size", type = int, default=500, help = "Size of feature space")
    command_parser.add_argument('-cg', "--clip_gradients", type = bool, default=False, help="Enable gradient clipping")
    command_parser.add_argument('-idx', "--idx_to_remove", type = int, default=40, help="Index of the last most common code to remove")
    command_parser.add_argument('-lr', "--learning_rate", type = float, default = 0.001, help = "Learning rate value")
    command_parser.add_argument('-alr', "--adaptive_lr", type = bool, default = False, help = "Enable adaptive lr")
    command_parser.add_argument('--hidden_size', type = int, default = 100, help = "RNNcell hidden state size")
    command_parser.add_argument('--max_length', type = int, default = 50, help = "Max length of code sequence to consider")
    command_parser.add_argument('--pos_weight', type = float, default = 1, help="Weight for positive examples in the loss")
    command_parser.set_defaults(func=do_evaluate)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

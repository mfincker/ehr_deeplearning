#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
RNN model for EHR project - adapted from CS224N
"""
from __future__ import division
import logging

import tensorflow as tf
from util import Progbar, minibatches
import numpy as np

from gru_cell import GRUCell
from rnn_cell import RNNCell

logger = logging.getLogger("RNNModel")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class RNNModel(object):
    """
    Implements special functionality for NER models.
    """
    def __init__(self, config, pretrained_embeddings = None, report=None):
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.index_placeholder = None


        self.config = config
        self.report = report

        self.build()

    def pad_sequences(self, data, max_length):
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

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        return self.pad_sequences(examples, self.config.max_length)
    
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, 1), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        """

        self.input_placeholder = tf.placeholder(tf.int32, shape = (None, self.config.max_length), name = "input_placeholder")
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (None), name = "labels_placeholder")
        self.index_placeholder = tf.placeholder(tf.int32, shape = (None), name = "index_placeholder")

    def create_feed_dict(self, inputs_batch, index_batch, labels_batch=None):
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
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {   self.input_placeholder: inputs_batch,
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
                                    name="one_hot_embedded_input",
                                ), tf.float32)
        else:
            embeddings = tf.Variable(self.pretrained_embeddings, dtype = tf.float32, name = "vocabulary", trainable=True)
            embeddings = tf.nn.embedding_lookup(params = embeddings, ids = self.input_placeholder)
            embeddings = tf.reshape(tensor = embeddings, shape = [-1, self.config.max_length, self.config.n_features * self.config.embed_size])                                                  
                   
        return embeddings

    def add_prediction_op(self):
        """

        Returns:
            pred: tf.Tensor of shape (batch_size, 1)
        """



        if self.config.cell == "rnn":
            cell = RNNCell(self.config.feature_size, self.config.hidden_size)
        elif self.config.cell == "gru_tf":
            cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(self.config.feature_size, self.config.hidden_size)
        elif self.config.cell == "lstm":
            cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
        elif self.config.cell == "multi_gru":
		rnn_layers = [GRUCell(f, h) for f, h in [(self.config.feature_size, 500), (500, 50), (50, self.config.hidden_size)]]
		cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        elif self.config.cell == "multi_rnn":
            rnn_layers = [tf.contrib.rnn.BasicRNNCell(size) for size in [300, 150, self.config.hidden_size]]
            cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        else:
            raise ValueError("Unsupported cell type.")

        x = self.add_embedding()

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size * self.config.max_length, self.config.embed_size])
        U0 = tf.get_variable(name = "U0", shape = [self.config.embed_size, self.config.feature_size], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(name = "b0", shape = [self.config.feature_size])
        x = tf.tanh(tf.matmul(x, U0) + b0)
        x = tf.reshape(x, shape = [batch_size, self.config.max_length, self.config.feature_size])


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

    def predict_on_batch(self, sess, inputs_batch, index_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, index_batch=index_batch)
        predictions = sess.run(tf.cast(tf.reshape(self.pred, [-1]) > 0, tf.int32 ), feed_dict=feed)
        # predictions = sess.run(self.pred, feed_dict=feed)

        return predictions

    def train_on_batch(self, sess, inputs_batch, index_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, index_batch=index_batch)
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
        return loss

    def evaluate(self, sess, x, y):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            x: A list of list of code indices / 
            y: A list of labels
        Returns:
            The F1 score for predicting tokens as named entities (and other metrics)
        """

        correct_preds, total_correct, total_preds = 0., 0., 0.
        labels, preds = self.output(sess, x, y)

        accuracy = sum([l == p for l, p in zip(labels, preds)]) / len(labels)
        precision = sum([l == p for l, p in zip(labels, preds) if l == 1]) / sum(preds)
        recall = sum([l == p for l, p in zip(labels, preds) if l == 1]) / sum(labels)
        f1 = 2 / ((1/precision) + (1/recall))

        return (accuracy, precision, recall, f1)


    def output(self, sess, x, y):
        """
        Reports the output of the model on examples.
        """

        inputs = self.preprocess_sequence_data(x)
        inputs = [(t[0], t[1], l) for t, l in zip(inputs, y)]

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        labels = []
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict

            batch_ = batch[:-1]
            labels_ = batch[-1]

            preds_ = self.predict_on_batch(sess, *batch_)

            preds.extend(list(preds_))
            labels.extend(list(labels_))
            prog.update(i + 1, [])
        return labels, preds

    def fit(self, session, saver, train_x, train_y, dev_x, dev_y):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_x)
        train_examples = [(t[0], t[1], l) for t, l in zip(train_examples, train_y)]
        
        dev_data = self.preprocess_sequence_data(dev_x)
        dev_data = [(t[0], t[1], l) for t, l in zip(dev_data, dev_y)]
        # logger.info(dev_data)

        epoch_losses = []
        for epoch in range(self.config.n_epochs):
            if self.config.adaptive_lr:
                pass
                # if epoch == 10:
                    # prev_lr = self.config.lr
                    # self.config.lr = 0.0001
                    # logger.info("Updating lr from %f to %f", [prev_lr, self.config.lr])
                
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
			
            batch_losses = []
            for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
                
                loss = self.train_on_batch(session, *batch)
                prog.update(i + 1, exact = [("train loss", loss)])
                batch_losses.append(loss)

            epoch_losses.append(batch_losses)

            #if epoch % 10 == 0:
            #    logger.info("Evaluating on training data")
            #    entity_scores = self.evaluate(session, train_x, train_y)
            #    logger.info("Accuracy/Precision/Recall/F1: %.2f/%.2f/%.2f/%.2f", *entity_scores)
            logger.info("Evaluating on development data")
            entity_scores = self.evaluate(session, dev_x, dev_y)
            logger.info("Accuracy/Precision/Recall/F1: %.2f/%.2f/%.2f/%.2f", *entity_scores)

            score = entity_scores[-1]
            
            if score >= best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(session, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()

        return best_score, epoch_losses

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)


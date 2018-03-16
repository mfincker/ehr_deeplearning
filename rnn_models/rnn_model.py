#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
RNN model for EHR project - adapted from CS224N
"""
import pdb
import logging

import tensorflow as tf
from util import Progbar, minibatches
import numpy as np


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("RNN")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class RNNModel(object):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, config, report=None):
        self.config = config
        self.report = report

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def evaluate(self, sess, x, y):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            x: A list of list of code indices
            y: A list of labels
        Returns:
            The F1 score for predicting tokens as named entities (and other metrics)
        """

        correct_preds, total_correct, total_preds = 0., 0., 0.
        labels, preds = self.output(sess, x, y)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

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
            neg_idx = np.sum(batch[1] < 0)

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

        epoch_losses = []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
			
            batch_losses = []
            for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
                
                loss = self.train_on_batch(session, *batch)
                prog.update(i + 1, exact = [("train loss", loss)])
                batch_losses.append(loss)

            epoch_losses.append(batch_losses)
            
            logger.info("Evaluating on training data")
            entity_scores = self.evaluate(session, train_x, train_y)
            logger.info("Accuracy/Precision/Recall/F1: %.2f/%.2f/%.2f/%.2f", *entity_scores)
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

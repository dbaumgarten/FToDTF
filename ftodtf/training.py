#!/usr/env python3

# Copyright 2018 The FToDTF Authors.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ATTENTION: the original file (found at https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
# has been completely modified by the FToDTF Authors!
#
#
# ==============================================================================
""" This module handles the training of the word-vectors"""
import os
import sys

import tensorflow as tf

import ftodtf.input as inp
import ftodtf.model as model
import ftodtf.validation

CURRENNT_PATH = os.getcwd()
DEFAULT_LOGPATH = os.path.join(CURRENNT_PATH, "log")


def train(corpus_path, log_dir=DEFAULT_LOGPATH,
          steps=100001,
          vocabulary_size=50000,
          batch_size=128,
          embedding_size=128,
          skip_window=1,
          num_sampled=64,
          ngram_size=3,
          num_buckets=100000,
          validation_words=None):
    """ Run the fasttext training. ATTENTION!: The default values for the log_dir reported by sphinx are wrong!
        The correct default is listed below.

         :param str corpus_path: Path to the corpus/ training_data.
         :param str log_dir: Directory to write the generated files to. Default: <current-dir>/log
         :param int steps: How many training steps to perform
         :param int vocabulary_size: How many words the vocabulary will have. Only the vocabulary_size most frequent words will be processed.
         :param int batch_size: How many trainings-samples to process per batch
         :param int embedding_size: Dimension of the computed embedding vectors.
         :param int skip_window: How many words to consider left and right of the target-word
         :param int num_sampled: Number of negative examples to sample when computing the nce_loss
         :param int ngram_size: How large the ngrams (in which the target words are split) should be
         :param int num_buckets: How many hash-buckets to use when hashing the ngrams to numbers
         :param list(str) validation_words: A list of words. The similarity of these words to each other will be regularily computed to indicade the training-progress
    """
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    try:
        inp.check_valid_path(corpus_path)

    except FileNotFoundError as e:
        print(": ".join(["ERROR", e.__str__()]))
        sys.exit(-1)  # EXIT

    else:
        filename = inp.unarchive(corpus_path)

    # Read the data into a list of strings.
    print("Reading dataset")
    p = inp.InputProcessor(filename, skip_window,
                           batch_size, vocabulary_size, ngram_size)
    p.preprocess()

    # Get the computation-graph and the associated operations
    m = model.Model(p.batches, batch_size, embedding_size,
                    vocabulary_size, validation_words, num_sampled, num_buckets)

    with tf.Session(graph=m.graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize the model before it can be used
        m.init()
        print('Initialized')

        average_loss = 0
        for step in range(steps):

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [m.optimizer, m.merged, m.loss],
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0 and validation_words:
                ftodtf.validation.print_similarity_check(
                    m.validation, validation_words)

        # Save the model for checkpoints.
        m.save(session, os.path.join(log_dir, 'model.ckpt'))

    writer.close()

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

# pylint: disable=E0611
from tensorflow.python.client import timeline


def train(settings):
    """ Run the fasttext training.

    :param settings: An object encapsulating all the settings for the fasttext-model
    :type settings: ftodtf.settings.FasttextSettings

    """
    print(settings.log_dir)
    if not os.path.exists(settings.log_dir):
        os.makedirs(settings.log_dir)

    try:
        inp.check_valid_path(settings.batches_file)

    except FileNotFoundError as e:
        print(": ".join(["ERROR", e.__str__()]))
        sys.exit(-1)  # EXIT

    # Get the computation-graph and the associated operations
    m = model.Model(settings)

    with tf.Session(graph=m.graph) as session:

        # try to restore the saved session if needed
        if settings.save_mode is True:
            m.restore(session, os.path.join(settings.log_dir,
                                            'model.ckpt'))
            print("Model restored.")

        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(settings.log_dir, session.graph)
        try:
            # We must initialize the model before it can be used
            m.init()
            print('Initialized')

            average_loss = 0
            for step in range(settings.steps):

                # Define metadata variable.
                run_metadata = tf.RunMetadata()
                options = None
                if settings.profile:
                    # pylint: disable=E1101
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [m.optimizer, m.merged, m.loss],
                    run_metadata=run_metadata,
                    options=options)
                average_loss += loss_val

                # Create the Timeline object, and write it to a json file
                if settings.profile:
                    # pylint: disable=E1101
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('profile.json', 'w') as f:
                        f.write(chrome_trace)

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (settings.steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0 and settings.validation_words:
                    ftodtf.validation.print_similarity_check(
                        m.validation, settings.validation_words_list)

        # Save the model if something happens
        finally:
            # Save the model for checkpoints.
            m.save(session, os.path.join(settings.log_dir, 'model.ckpt'))
            writer.close()

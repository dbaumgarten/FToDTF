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
# has been modified by the FToDTF Authors!
#
#
# ==============================================================================

import os
import sys

import tensorflow as tf
import numpy as np

import ftodtf.input as inp
import ftodtf.model as model
import ftodtf.evaluation as evaluation

current_path = os.getcwd()
default_logpath = os.path.join(current_path,"log")


def run(corpus_path, log_dir=default_logpath,
        steps = 100001,
        vocabulary_size = 50000,
        batch_size = 128,
        embedding_size = 128,
        skip_window = 1,
        num_sampled = 64,
        valid_size = 16,
        valid_window = 100):
  """ Run the fasttext training. ATTENTION!: The default values for the log_dir reported by sphinx are wrong!
      The correct default is listed below.

       :param str corpus: Path to the corpus/ training_data.
       :param str log_dir: Directory to write the generated files to. Default: <current-dir>/log
       :param int steps: How many training steps to perform
       :param int vocabulary_size: How many words the vocabulary will have. Only the vocabulary_size most frequent words will be processed.
       :param int batch_size: How many trainings-samples to process per batch
       :param int embedding_size: Dimension of the computed embedding vectors.
       :param int skip_window: How many words to consider left and right of the target-word
       :param int num_sampled: Number of negative examples to sample when computing the nce_loss
       :param int valid_size: Number of random words to use for validation
       :param int valid_window: Choose random valid_saze words from the valid_window most frequend words to use for validation
  """
  print(log_dir)
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)

  try:
    inp.check_valid_path(corpus_path)

  except FileNotFoundError as e:
    print(": ".join(["ERROR", e.__str__()]))
    sys.exit(-1) # EXIT

  else:
    filename = inp.unarchive(corpus_path)

  # Read the data into a list of strings.
  print("Reading dataset")
  p = inp.InputProcessor(filename,1,128,vocabulary_size)
  p.preprocess()

  reverse_dictionary = p.reversed_dict
  # Step 4: Build and train a skip-gram model.

  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent. These 3 variables are used only for
  # displaying model accuracy, they don't affect calculation.
  # pylint: disable=no-member
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  # Get the computation-graph and the associated operations
  m = model.Model(p.batches,batch_size,embedding_size,vocabulary_size,valid_examples,num_sampled)


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
          [m.optimizer,m.merged,m.loss],
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
      if step % 10000 == 0:
        evaluation.printSimilarityCheck(m.similarity,valid_examples,valid_size,reverse_dictionary)

    final_embeddings = m.normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in range(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    m.save(session, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    evaluation.tensorboardVisualisation(m.embeddings,writer,log_dir)

  writer.close()

  # Step 6: Visualize the embeddings.
  evaluation.visualizeEmbeddings(reverse_dictionary,final_embeddings,log_dir)

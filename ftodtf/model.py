"""This module handles the building of the tf execution graph"""
import tensorflow as tf
import math


class Model():
    def __init__(self,batch_size,embedding_size,vocabulary_size,valid_examples,num_sampled):
        """Builds and returns the computation graph. The Model-Object has properties representing the inputs, the graph and the output-operations"""
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=self.train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', self.loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,valid_dataset)
            self.similarity = tf.matmul(
                valid_embeddings, self.normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

            # Create a saver.
            self.saver = tf.train.Saver()

    def save(self,session,file):
        return self.saver.save(session,file)
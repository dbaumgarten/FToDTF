"""This module handles the building of the tf execution graph"""
import tensorflow as tf
import math


class Model():
    """Builds and represents the tensorflow computation graph. Exports all important operations via fields"""
    def __init__(self,sample_generator_func,batch_size,embedding_size,vocabulary_size,valid_examples,num_sampled):
        """
        Constuctor for Model

        :param sample_generator_func: A function returning a generator yielding training-batches. See: ftodtf.input.InputProcessor.batches
        :param int batch_size: The size the trainings-batches (obtained via sample_generator_func) will have
        :param int embedding_size: The size of the word-embedding-vectors to generate
        :param int vocabulary_size: How many words are in the vocabulary.
        :param list(int) valid_examples: A list of ints each representing a word from the vocabulary. These words will be used when validating the trained model
        :param int num_sampled: How many negative samples to draw during nce_loss calculation
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            #create a dataset pipeline from the given sample-generator
            inputpipe = tf.data.Dataset.from_generator(sample_generator_func,output_types=(tf.int32,tf.int32),output_shapes=([batch_size],[batch_size,1]))
            inputpipe = inputpipe.prefetch(1)
            iterator = inputpipe.make_initializable_iterator()
            self.dataset_init = iterator.initializer
            batch = iterator.get_next()

            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = batch[0]
                self.train_labels = batch[1]
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
            self.variable_init = tf.global_variables_initializer()

            # Create a saver.
            self.saver = tf.train.Saver()

    def save(self,session,file):
        """Save the current session (including variable values) to a file

        :param session: The current session to save
        :type session: tf.Session
        :param str file: The path to save to
        :returns: The result of saver.save()
        """
        return self.saver.save(session,file)

    def init(self):
        """ Initialize variables and the input iterator.

            Must be called before everything else.
            Must be called inside a tf.Session
        """
        self.variable_init.run()
        self.dataset_init.run()
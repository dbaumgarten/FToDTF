"""This module handles the building of the tf execution graph"""
import math
import tensorflow as tf
import ftodtf.input as inp


class Model():
    """Builds and represents the tensorflow computation graph. Exports all important operations via fields"""

    def __init__(self, sample_generator_func, settings):
        """
        Constuctor for Model

        :param settings: An object encapsulating all the settings for the fasttext-model
        :type settings: ftodtf.settings.FasttextSettings
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            # create a dataset pipeline from the given sample-generator
            inputpipe = tf.data.Dataset.from_generator(sample_generator_func, output_types=(
                tf.string, tf.int32), output_shapes=([settings.batch_size, None], [settings.batch_size, 1]))
            inputpipe = inputpipe.prefetch(1)
            iterator = inputpipe.make_initializable_iterator()
            self._dataset_init = iterator.initializer
            batch = iterator.get_next()

            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = batch[0]
                train_labels = batch[1]

            # Create all Weights
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform(
                    [settings.num_buckets-1, settings.embedding_size], -1.0, 1.0))
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([settings.vocabulary_size, settings.embedding_size], stddev=1.0 / math.sqrt(settings.embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([settings.vocabulary_size]))

            # The vector for the placeholder-ngram (""). Will always be <0...> and therefore be irrelevant for reduce_sum
            padding_vector = tf.constant(
                0.0, shape=[1, settings.embedding_size])
            # Set the first enty in embeddings (belonging to the padding-ngram) to <0,0,...>
            self.embeddings = tf.concat([padding_vector, self.embeddings], 0)

            # Bucket-hash the ngrams. Make sure "" is always hashed to 0
            # pylint: disable=E1101
            self._hasher = tf.contrib.lookup.index_table_from_tensor(
                [""], num_oov_buckets=settings.num_buckets-1, hasher_spec=tf.contrib.lookup.FastHashSpec, dtype=tf.string)

            target_vectors = self._ngrams_to_vectors(train_inputs)

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=target_vectors,
                        num_sampled=settings.num_sampled,
                        num_classes=settings.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', self.loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(
                    1).minimize(self.loss)

            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Create a saver to save the trained variables once training is over
            self._saver = tf.train.Saver()

            if settings.validation_words:
                self.validation = self._validationop(
                    settings.validation_words_list)

    def _ngrams_to_vectors(self, ngrams):
        """ Convert a batch consisting of lists of ngrams for a word to a list of vectors. One vector for each word

        :param ngrams: A batch of lists of ngrams
        :returns: a batch of vectors
        """

        hashed = self._hasher.lookup(ngrams)
        # Lookup the vector for each hashed value. The hash-value 0 (the value for the ngram "") will always et a 0-vector
        looked_up = tf.nn.embedding_lookup(self.embeddings, hashed)
        # sum all ngram-vectors to get a word-vector
        summed = tf.reduce_sum(looked_up, 1)
        return summed

    def _validationop(self, compare):
        """This Operation is used to regularily computed the words closest to some input words. This way a human can judge if the training is really making usefull progress

        :param list(str) compare: A list of strings representing the words to find similar words of
        """

        # ngrammize and pad the words
        ngrams = [inp.generate_ngram_per_word(x) for x in compare]
        maxlen = 0
        for ng in ngrams:
            maxlen = max(maxlen, len(ng))
        for i, _ in enumerate(ngrams):
            ngrams[i] = inp.pad_to_length(ngrams[i], maxlen)

        dataset = tf.constant(ngrams, dtype=tf.string,
                              shape=[len(compare), maxlen])
        vectors = self._ngrams_to_vectors(dataset)

        # normalize word-vectors before computing dot-product (so the results stay between -1 and 1)
        norm = tf.sqrt(tf.reduce_sum(
            tf.square(vectors), 1, keep_dims=True))
        normalized_embeddings = vectors / norm

        return tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

    def save(self, session, file):
        """Save the current session (including variable values) to a file

        :param session: The current session to save
        :type session: tf.Session
        :param str file: The path to save to
        :returns: The result of saver.save()
        """
        return self._saver.save(session, file)

    def init(self):
        """ Initialize variables and the input iterator.

            Must be called before everything else.
            Must be called inside a tf.Session
        """
        tf.global_variables_initializer().run()
        self._dataset_init.run()
        tf.tables_initializer().run()

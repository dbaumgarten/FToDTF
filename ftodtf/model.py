"""This module handles the building of the tf execution graph"""
import math
import tensorflow as tf
import ftodtf.input as inp


def parse_batch_func(batch_size):
    """ Returns a function that can parse a batch from a tfrecord-entry

    :param int batch_size: How many samples are in a batch
    """
    def parse(batch):
        """ Parses a tfrecord-entry into a usable batch. To be used with tf.data.Dataset.map

        :params batch: The tfrecord-entry to parse
        :returns: A batch ready to feed into the model
        """
        features = {
            "inputs": tf.VarLenFeature(tf.int64),
            "labels": tf.FixedLenFeature([batch_size], tf.int64)
        }
        parsed = tf.parse_single_example(batch, features=features)
        inputs = tf.sparse_tensor_to_dense(
            parsed['inputs'], default_value=0)
        inputs = tf.reshape(inputs, [batch_size, -1])
        labels = tf.reshape(parsed["labels"], [batch_size, 1])
        return inputs, labels
    return parse


class Model():
    """Builds and represents the tensorflow computation graph. Exports all important operations via fields"""

    def __init__(self, settings, cluster=None):
        """
        Constuctor for Model

        :param settings: An object encapsulating all the settings for the fasttext-model
        :param cluster: A tf.train.ClusterSpec object describint the tf-cluster. Needed for variable and ops-placement
        :type settings: ftodtf.settings.FasttextSettings
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            device = None
            if cluster and settings.ps_list:  # if running distributed use replica_device_setter
                device = tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % settings.index, cluster=cluster)
            # If running distributed pin all ops and assign variables to ps-servers. Else use auto-assignment
            with tf.device(device):

                inputpipe = tf.data.TFRecordDataset(
                    [settings.batches_file]).repeat()
                batches = inputpipe.map(parse_batch_func(
                    settings.batch_size), num_parallel_calls=4)
                batches = batches.prefetch(1)

                iterator = batches.make_initializable_iterator()
                self._dataset_init = iterator.initializer
                batch = iterator.get_next()

                # Input data.
                with tf.name_scope('inputs'):
                    train_inputs = batch[0]
                    train_labels = batch[1]

                # Create all Weights
                with tf.name_scope('embeddings'):
                    self.embeddings = tf.Variable(tf.random_uniform(
                        [settings.num_buckets, settings.embedding_size], -1.0, 1.0))
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal([settings.vocabulary_size, settings.embedding_size], stddev=1.0 / math.sqrt(settings.embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(
                        tf.zeros([settings.vocabulary_size]))

                # Set the first enty in embeddings (belonging to the padding-ngram) to <0,0,...>
                self.mask_padding_zero_op = tf.scatter_update(
                    self.embeddings, 0, tf.zeros([settings.embedding_size], dtype=tf.float32))

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

                # Keep track of how many iterations we have already done
                self.step_nr = tf.train.create_global_step(self.graph)

                # Learnrate starts at settings.learnrates and will reach ~0 when the training is finished.
                decaying_learn_rate = settings.learnrate * \
                    (1 - (self.step_nr/settings.steps))

                # Add the learnrate to the summary
                tf.summary.scalar('learnrate', decaying_learn_rate)

                with tf.name_scope('optimizer'):
                    self.optimizer = tf.train.GradientDescentOptimizer(
                        decaying_learn_rate).minimize(self.loss, global_step=self.step_nr)

                # Merge all summaries.
                self.merged = tf.summary.merge_all()

                # Create a saver to save the trained variables once training is over
                self._saver = tf.train.Saver()

                if settings.validation_words:
                    self.validation = self._validationop(
                        settings.validation_words_list, settings.num_buckets)

    def _ngrams_to_vectors(self, ngrams):
        """ Convert a batch consisting of lists of ngrams for a word to a list of vectors. One vector for each word

        :param ngrams: A batch of lists of ngrams
        :returns: a batch of vectors
        """
        # Lookup the vector for each hashed value. The hash-value 0 (the value for the ngram "") will always et a 0-vector
        with tf.control_dependencies([self.mask_padding_zero_op]):
            looked_up = tf.nn.embedding_lookup(self.embeddings, ngrams)
            # sum all ngram-vectors to get a word-vector
            summed = tf.reduce_sum(looked_up, 1)
            return summed

    def _validationop(self, compare, num_buckets):
        """This Operation is used to regularily computed the words closest to some input words. This way a human can judge if the training is really making usefull progress

        :param list(str) compare: A list of strings representing the words to find similar words of
        :param int num_buckets: The number of hash-buckets used when hashing ngrams
        """

        # ngrammize and pad the words
        ngrams = [inp.generate_ngram_per_word(x) for x in compare]
        maxlen = 0
        for ng in ngrams:
            maxlen = max(maxlen, len(ng))
        for i, _ in enumerate(ngrams):
            ngrams[i] = inp.hash_string_list(ngrams[i], num_buckets-1, 1)
            ngrams[i] = inp.pad_to_length(ngrams[i], maxlen, pad=0)

        dataset = tf.constant(ngrams, dtype=tf.int64,
                              shape=[len(compare), maxlen])
        vectors = self._ngrams_to_vectors(dataset)

        # normalize word-vectors before computing dot-product (so the results stay between -1 and 1)
        norm = tf.sqrt(tf.reduce_sum(
            tf.square(vectors), 1, keep_dims=True))
        normalized_embeddings = vectors / norm

        return tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

    def get_scaffold(self):
        """ Returns a tf.train.Scaffold object describing this graph

        :returns: tf.train.Scaffold
        """
        return tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.group(tf.local_variables_initializer(
            ), self._dataset_init, tf.tables_initializer()),
            saver=self._saver,
            summary_op=self.merged
        )

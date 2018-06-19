"""This module handles the building of the tf execution graph"""
import tensorflow as tf
import math
import ftodtf.input as inp


class Model():
    """Builds and represents the tensorflow computation graph. Exports all important operations via fields"""
    def __init__(self,sample_generator_func,batch_size,embedding_size,vocabulary_size,validation_words,num_sampled,num_buckets):
        """
        Constuctor for Model

        :param sample_generator_func: A function returning a generator yielding training-batches. See: ftodtf.input.InputProcessor.batches
        :param int batch_size: The size the trainings-batches (obtained via sample_generator_func) will have
        :param int embedding_size: The size of the word-embedding-vectors to generate
        :param int vocabulary_size: How many words are in the vocabulary.
        :param list(str) validation_words: A list of words. The similarity between these words can than be computed using the self.similarity operation.
        :param int num_sampled: How many negative samples to draw during nce_loss calculation
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            #create a dataset pipeline from the given sample-generator
            inputpipe = tf.data.Dataset.from_generator(sample_generator_func,output_types=(tf.string,tf.int32),output_shapes=([batch_size,None],[batch_size,1]))
            inputpipe = inputpipe.prefetch(1)
            iterator = inputpipe.make_initializable_iterator()
            self.dataset_init = iterator.initializer
            self.batch = iterator.get_next()

            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = self.batch[0]
                self.train_labels = self.batch[1]

            # Create all Weights
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform([num_buckets, embedding_size], -1.0, 1.0))
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Hash the input strings to int-buckets
            self.hashed = tf.string_to_hash_bucket_fast(self.train_inputs,num_buckets)
            # look up the corresponding vector for each hash
            self.looked_up = tf.nn.embedding_lookup(self.embeddings, self.hashed)
            # sum up all n-gram vectors of a word to get a vector for the whole word
            self.target_vectors = tf.reduce_sum(self.looked_up,1)

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=self.train_labels,
                        inputs=self.target_vectors,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', self.loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            # Only used once at the end of the training just before saving the weights
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
 
            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer.
            self.variable_init = tf.global_variables_initializer()

            # Create a saver to save the trained variables once training is over
            self.saver = tf.train.Saver()

            if validation_words:
                self.similarity = self._validationop(validation_words,self.normalized_embeddings,num_buckets)

            

    def _validationop(self,compare,embeddings,num_buckets):
        """This Operation is used to regularily computed the words closest to some input words. This way a human can judge if the training is really making usefull progress
        
        :param list(str) compare: A list of strings representing the words to find similar words of
        :param tf.Tensor embeddings: The trained vector-representation to use for the computation
        :param int num_buckets: How many buckets to use when hashing the words. Must be the same value as used when computing the embeddings
        """

        # ngrammize and pad the words
        ngrams = [ inp.generate_ngram_per_word(x) for x in compare]
        maxlen = 0
        for ng in ngrams:
            maxlen = max(maxlen,len(ng))
        for i in range(len(ngrams)):
            ngrams[i] = inp.pad_to_length(ngrams[i],maxlen)

        valid_dataset = tf.constant(ngrams, dtype=tf.string, shape=[len(compare),maxlen])
        valid_embeddings = tf.string_to_hash_bucket_fast(valid_dataset,num_buckets)
        valid_embeddings = tf.nn.embedding_lookup(embeddings,valid_embeddings)
        valid_embeddings = tf.reduce_sum(valid_embeddings,1)
        return tf.matmul(valid_embeddings, valid_embeddings, transpose_b=True)


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
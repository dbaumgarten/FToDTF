""" This module contains the FasttextSettings class """
import os
import re

CURRENNT_PATH = os.getcwd()
DEFAULT_LOGPATH = os.path.join(CURRENNT_PATH, "log")
DEFAULT_BATCHES_FILE = os.path.join(CURRENNT_PATH, "batches.tfrecord")

# pylint: disable=R0902,R0903


class FasttextSettings:
    """ This class contains all the settings for the fasttext-training and also handles things like validation. Use the attributes/variables of this class to set hyperparameters for the model.

    :ivar str corpus_path: Path to the file containing text for training the model.
    :ivar str batches_file: The Filename for the file containing the training-batches. The file is written by the preprocess command and read by the train command.
    :ivar str log_dir: Directory to write the generated files (e.g. the computed word-vectors) to.
    :ivar int steps: How many training steps to perform.
    :ivar int vocabulary_size: How many words the vocabulary will have. Only the vocabulary_size most frequent words will be processed.
    :ivar int batch_size: How many trainings-samples to process per batch.
    :ivar int embedding_size: Dimension of the computed embedding vectors.
    :ivar int skip_window: How many words to consider left and right of the target-word.
    :ivar int num_sampled: Number of negative examples to sample when computing the nce_loss.
    :ivar int ngram_size: How large the ngrams (in which the target words are split) should be.
    :ivar int num_buckets: How many hash-buckets to use when hashing the ngrams to numbers.
    :ivar str validation_words: A string of comma-seperated words. The similarity of these words to each other will be regularily computed and printed to indicade the progress of the training.
    :ivar boolean profile: If set to True tensorflow will profile the graph-execution and writer results to ./profile.json.
    :ivar float learnrate: The starting learnrate for the training. The actual learnrate will lineraily decrease to beyth 0 when the specified amount of training-steps is reached.
    :ivar float rejection_threshold: In order to subsample the most frequent words.
    :ivar string job: The role of this node in a distributed setup. Can be worker' or 'ps'.
    :ivar str workers: A comma seperated list of host:port combinations representing the workers in the distributed setup.
    :ivar str ps: A comma seperated list of host:port combinations representing the parameter servers in the distributed setup. If empty a non-distributed setup is assumed.
    :ivar int index: The of the node itself in the list of --workers (or --ps, depending on --job).
    """

    def __init__(self):
        self.corpus_path = ""
        self.batches_file = DEFAULT_BATCHES_FILE
        self.log_dir = DEFAULT_LOGPATH
        self.steps = 100001
        self.vocabulary_size = 50000
        self.batch_size = 128
        self.embedding_size = 300
        self.skip_window = 1
        self.num_sampled = 5
        self.ngram_size = 3
        self.num_buckets = 10000   # In paper 210**6, but "test_fasttext" will fail
        self.validation_words = ""
        self.profile = False
        self.learnrate = 0.1
        self.rejection_threshold = 0.0001
        self.job = "worker"
        self.index = 0
        self.workers = "localhost:7777"
        self.ps = ""

    @staticmethod
    def preprocessing_settings():
        """
        Returns the names of the settings that are used for the preprocessing
        command
        """
        return ["corpus_path", "batches_file", "vocabulary_size",
                "batch_size", "skip_window", "ngram_size", "num_buckets",
                "rejection_threshold", "profile"]

    @staticmethod
    def training_settings():
        """ Returns the names of the settings that are used for the training
        command """
        return ["batches_file", "log_dir", "steps", "vocabulary_size",
                "batch_size", "embedding_size", "num_sampled", "num_buckets",
                "validation_words", "profile", "learnrate", "save_mode"]

    @staticmethod
    def distribution_settings():
        """ Returns the names of the settings that are used for configuren the tensoflow-cluster """
        return ["job", "index", "workers", "ps"]

    @property
    def validation_words_list(self):
        """ Returns the validation_words as list of strings instead of a comma
        seperate string like the attribute would do
        :returns: A list of strings if validation_words is set and else None
        """
        if self.validation_words:
            return self.validation_words.split(",")
        return None

    @property
    def workers_list(self):
        """ Returns workers as list of strings instead of a comma
        seperate string like the attribute would do
        :returns: A list of strings if workers is set and else None
        """
        if self.workers:
            return self.workers.split(",")
        return []

    @property
    def ps_list(self):
        """ Returns ps as list of strings instead of a comma
        seperate string like the attribute would do
        :returns: A list of strings if ps is set and else None
        """
        if self.ps:
            return self.ps.split(",")
        return []

    def validate_preprocess(self):
        """ Check if the current settings are valid for pre processing.
        :raises: ValueError if the validation fails"""
        try:
            check_corpus_path(self.corpus_path)
            check_vocabulary_size(self.vocabulary_size)
            check_batch_size(self.batch_size)
            check_skip_window(self.skip_window)
            check_ngram_size(self.ngram_size)
            check_num_buckets(self.num_buckets)
            check_rejection_threshold(self.rejection_threshold)
        except Exception as e:
            raise e

    def validate_train(self):
        """Check if the current settings are valid for training.
        :raises: ValueError if the validation fails """
        try:
            if self.job != "ps":
                check_batches_file(self.batches_file)
            if self.index == 0 and self.job == "worker":
                check_log_dir(self.log_dir)
            check_steps(self.steps)
            check_vocabulary_size(self.vocabulary_size)
            check_batch_size(self.batch_size)
            check_embedding_size(self.embedding_size)
            check_num_sampled(self.num_sampled)
            check_num_buckets(self.num_buckets)
            check_learn_rate(self.learnrate)
            check_nodelist(self.workers)
            check_nodelist(self.ps, allow_empty=True)
            check_job(self.job)
            check_index(self.job, self.workers, self.ps, self.index)
        except Exception as e:
            raise e

    def attribute_docstring(self, attribute, include_defaults=True):
        """ Given the name of an attribute of this class, this function will return the docstring for the attribute.

        :param str attribute: The name of the attribute
        :returns: The docstring for the attribute
        """
        match = re.search("^.*:ivar \\w* "+attribute +
                          ": (.*)$", self.__doc__, re.MULTILINE)
        if not match:
            raise RuntimeError("No docstring found for: "+attribute)
        docstring = match.group(1)
        if include_defaults:
            docstring += " Default: "+str(vars(self)[attribute])

        return docstring


def check_index(job, workers, ps, index):
    if job == "worker":
        li = workers
    else:
        li = ps
    if index < 0 or index >= len(li.split(",")):
        raise ValueError(
            "--index must be between 0 and {}".format(len(li.split(","))))


def check_job(job):
    if job != "worker" and job != "ps":
        raise ValueError("--job can only be 'worker' or 'ps'")


def check_nodelist(noli, allow_empty=False):
    """ Checks if the given argument is a comma seperated list of host:port strings.

    :raises: ValueError if it is not
    """
    if allow_empty and noli == "":
        return
    hostportregex = re.compile("^[0-9a-zA-Z.]+:[0-9]+$")
    noli = noli.split(",")
    for e in noli:
        if not hostportregex.match(e):
            raise ValueError(
                "{} is not a valid host:port combination".format(e))


def check_corpus_path(corpus_path):
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError("The specified corpus was not found!")


def check_vocabulary_size(vocabulary_size):
    if vocabulary_size <= 0:
        raise ValueError("Vocabulary size must be bigger than zero.")
    elif vocabulary_size > 10251098:  # Number of English words --> biggest vocab
        raise ValueError("There exist no language with such a big vocabulary.")


def check_rejection_threshold(rejection_threshold):
    if rejection_threshold <= 0 or rejection_threshold > 1:
        raise ValueError("Rejection threshold must be between 0 and 1.")


def check_batch_size(batch_size):
    if batch_size < 32 or batch_size > 1024:
        # Practical recommendations for gradient-based training of deep architectures
        # https://arxiv.org/abs/1206.5533
        raise ValueError("The recommended batch-size should be between 32 and"
                         "1024")


def check_skip_window(skip_window):
    if skip_window < 1 or skip_window > 5:
        raise ValueError("The recommended window size should be between 1 and "
                         "5")


def check_ngram_size(ngram_size):
    if ngram_size < 3 or ngram_size > 6:
        raise ValueError("The recommended n-gram size should be between 3 and "
                         "6.")


def check_num_buckets(number_buckets):
    if number_buckets < 1:
        raise ValueError("Number of Buckets must be bigger than zero.")


def check_batches_file(batches_file):
    if not os.path.isfile(batches_file):
        raise FileNotFoundError(
            "The specified batches-file could not be found.")


def check_log_dir(log_dir):
    if not os.path.exists(log_dir):
        raise FileNotFoundError("Cannot find the log folder!")


def check_steps(steps):
    if steps < 1:
        raise ValueError("Number of steps should be bigger than 0.")


def check_embedding_size(embedding_size):
    if embedding_size < 50 or embedding_size > 1000:
        raise ValueError("The recommended embedding size should be between 50 "
                         "and 1000.")


def check_num_sampled(num_sampled):
    if num_sampled < 1 or num_sampled > 10:
        raise ValueError("The recommended number of negative samples should be"
                         "between 1 and 10.")


def check_learn_rate(learnrate):
    if learnrate < 0.01 or learnrate > 1.0:
        # https://fasttext.cc/docs/en/supervised-tutorial.html
        raise ValueError("The recommended learning rate should be between 0.01"
                         " and 1.0.")

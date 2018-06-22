""" This module contains the FasttextSettings class """
import os
import re


CURRENNT_PATH = os.getcwd()
DEFAULT_LOGPATH = os.path.join(CURRENNT_PATH, "log")

# pylint: disable=R0902,R0903


class FasttextSettings():
    """ This class contains all the settings for the fasttext-training and also handles things like validation. Use the attributes/variables of this class to set hyperparameters for the model.

    :ivar str corpus_path: Path to the file containing text for training the model.
    :ivar str log_dir: Directory to write the generated files to. Default: <current-dir>/log.
    :ivar int steps: How many training steps to perform.
    :ivar int vocabulary_size: How many words the vocabulary will have. Only the vocabulary_size most frequent words will be processed.
    :ivar int batch_size: How many trainings-samples to process per batch.
    :ivar int embedding_size: Dimension of the computed embedding vectors.
    :ivar int skip_window: How many words to consider left and right of the target-word.
    :ivar int num_sampled: Number of negative examples to sample when computing the nce_loss.
    :ivar int ngram_size: How large the ngrams (in which the target words are split) should be.
    :ivar int num_buckets: How many hash-buckets to use when hashing the ngrams to numbers.
    :ivar str validation_words: A string of comma-seperated words. The similarity of these words to each other will be regularily computed and printed to indicade the progress of the training.
    """

    def __init__(self):
        self.corpus_path = ""
        self.log_dir = DEFAULT_LOGPATH
        self.steps = 100001
        self.vocabulary_size = 50000
        self.batch_size = 128
        self.embedding_size = 128
        self.skip_window = 1
        self.num_sampled = 64
        self.ngram_size = 3
        self.num_buckets = 100000
        self.validation_words = ""

    @property
    def validation_words_list(self):
        """ Returns the validation_words as list of strings instead of a comma seperate string like the attribute would do

        :returns: A list of strings if validation_words is set and else None
        """
        if self.validation_words:
            return self.validation_words.split(",")
        return None

    def validate(self):
        """ Check if the current settings are valid

        :raises: ArgumentError of the validation fails
        """
        pass

    def attribute_docstring(self, attribute):
        """ Given the name of an attribute of this class, this function will return the docstring for the attribute.

        :param str attribute: The name of the attribute
        :returns: The docstring for the attribute
        """
        return re.search("^.*:ivar \\w* "+attribute+": (.*)$", self.__doc__, re.MULTILINE).group(1)

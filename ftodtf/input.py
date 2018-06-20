"""This module handles all the input-relatet tasks like loading, pre-processing and batching"""
import os
import zipfile
import tarfile
import collections
import random

from tempfile import gettempdir
from nltk import ngrams


def check_valid_path(file_path):
    """
    Checks if the given path exists.

    :param str path: The path to a file.
    :raises: FileNotFoundError
    """
    if os.path.isfile(file_path) is False:
        raise FileNotFoundError("The specified corpus was not found!")


def unarchive(file_path):
    """
    Checks if the given file is archived as tar or zips and calls the appropriate
    unarchive functions.

    :param str file_path: The path to a file.
    :returns: The path to the file after unarchiving.
    """
    if tarfile.is_tarfile(file_path):
        print("Unpacking tar ...")
        return untar_file(file_path)
    elif zipfile.is_zipfile(file_path):
        print("Unpacking zip ...")
        return unzip_file(file_path)
    else:
        return file_path


def unzip_file(file_path):
    """
    Extracts zip archive.

    :param str file_path: The path provided to the zip archive.
    :returns: The path to the extracted file.
    """
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extract(zip_ref.namelist()[0], path=gettempdir())
    file_name = os.path.join(gettempdir(), zip_ref.namelist()[0])
    zip_ref.close()
    return file_name


def untar_file(file_path):
    """
    Extracts tar archive.

    :param str file_path: The path provided to the
    :returns: The path to the extracted file.
    """
    tar_ref = tarfile.open(file_path)
    tar_ref.extractall(path=gettempdir())
    file_name = os.path.join(gettempdir(), tar_ref.getnames()[0])
    print(file_name)
    tar_ref.close()
    return file_name


def generate_ngram_per_word(word, ngram_window=2):
    """
    Generates ngram strings of the specified size for a given word.
    Before processing beginning and end of the word will be marked with "*".
    The ngrams will also include the full word (including the added *s).
    This is the same process as described in the fasttext paper.

    :param str word: The token string which represents a word.
    :param int ngram_window: The size of the ngrams
    :returns: A generator which yields ngrams.
    """
    word = "*"+word+"*"
    ngs = ngrams(word, ngram_window)
    ngstrings = ["".join(x) for x in ngs]
    ngstrings.append(word)
    return ngstrings


def pad_to_length(li, length, pad=""):
    """ Pads a given list to a given length with a given padding-element

    :param list() li: The list to be padded
    :param int length: The length to pad the list to
    :param object pad: The element to add to the list until the desired length is reached
    """
    li += [pad]*(length-len(li))
    return li


class InputProcessor():
    """Handles the creation of training-examble-batches from the raw training-text"""

    def __init__(self, filename, skip_window, batch_size, vocab_size, ngram_size):
        """
        Constructor of InputProcessor

        :param str filename: The path+name to the file to be used as input-data
        :param int skip_window: How many words left and right of a target word are considered as context words [skip_window][target_word][skip_window]
        :param int batch_size: How large each Training-batch should be
        :param int vocab_size: How large the vocabulary should be. Only the vocab_size most frequent words will be used for the vocabulary. The rest will be ignored.
        """
        self.filename = filename
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        # Will be populated by preprocess
        self.wordcount = None
        self.dict = None

    def preprocess(self):
        """ Do the needed proprocessing of the dataset. Count word frequencies, create a mapping word->int"""
        self.wordcount = collections.Counter(self._words_in_file())
        idx = 0
        self.dict = {}
        # Assign a number to every word we have. 0 = the most common word
        for word, _ in self.wordcount.most_common():
            # We only want vocab_size words in or dictionary. Skip the remaining uncommon words
            if idx == self.vocab_size:
                break
            self.dict[word] = idx
            idx += 1

    def _words_in_file(self):
        """Returns a generator over all words in the file"""
        with open(self.filename) as f:
            for line in f:
                words = line.split()
                for word in words:
                    yield word

    def string_samples(self):
        """ Returns a generator for samples (targetword->contextword)

        :returns: A generator yielding 2-tuple consisting of a target-word and a context word.
        """
        # possible positions of context-words relative to a target word
        contextoffsets = [
            x for x in range(-self.skip_window, self.skip_window+1) if x != 0]
        with open(self.filename) as f:
            for line in f:
                idx = 0
                words = line.split()
                for word in words:
                    # choose a random context word. Take special care to stay in the bounds of the list
                    contextoffset = random.choice(contextoffsets)
                    contextindex = idx+contextoffset
                    # if selected index-offset reaches outside of the list, try the other direction
                    if idx+contextoffset < 0 or idx+contextoffset >= len(words):
                        contextoffset = contextoffset*-1
                        # above fails if the current line is to short. Stay inside the bounds at all cost!
                        contextindex = min(
                            len(words)-1, max(0, idx+contextoffset))
                    yield (word, words[contextindex])
                    idx += 1

    def _lookup_label(self, gen):
        """ Maps the second words in the input-tuple to numbers.
            Conversion is done via lookup in self.dict

            :param gen: A generator yielding 2-tuples of strings
            :returns: A generator yielding 2-tuples (string,int)
        """
        for e in gen:
            try:
                yield (e[0], self.dict[e[1]])
            except KeyError:
                pass

    @staticmethod
    def _repeat(generator_func):
        """ Repeat a given generator forever by recreating it whenever a StopIteration Exception occurs

            :param generator_func: A function without arguments returning a generator
            :returns: A new inifinite generator
        """
        g = generator_func()
        while True:
            try:
                yield g.__next__()
            except StopIteration:
                g = generator_func()

    def _batch(self, samples):
        """ Pack self.batch_size of training samples into a batch
            The output is a tuple of two lists, rather then a list of tuples, because this way we can treat
            the two lists as input-tensor and label-tensor.
            The second list is al list of one-element-lists, because tf.nce_loss wants its tensor in that shape

            :param samples: A generator yielding 2-tuples
            :returns: A generator yielding 2-tuples of self.batch_size long lists. The second lists consists of 1-element-ling lists.
            """
        while True:
            inputs = []
            labels = []
            for _ in range(0, self.batch_size):
                sample = samples.__next__()
                inputs.append(sample[0])
                labels.append([sample[1]])
            yield inputs, labels

    def _ngrammize(self, gen):
        """ Transforms the first entry (a string) of the tuples received from the generator gen into a list of ngrams

        :param gen: A generator yielding tuples (str,?)
        :returns: A generator yielding tuples (list(str),?)
        """
        for entry in gen:
            yield (generate_ngram_per_word(entry[0], self.ngram_size), entry[1])

    @staticmethod
    def _equalize_batch(gen):
        """ Makes sure all n-gram arrays of a batch have the same length. 

        :param gen: The generator to retrieve the batches from
        :returns: A generator yielding batches with equal-length ngram-lists
        """
        for batch in gen:
            longest = 0
            for ngs in batch[0]:
                longest = max(longest, len(ngs))
            for i in range(len(batch[0])):
                batch[0][i] = pad_to_length(batch[0][i], longest)
            yield batch

    def batches(self):
        """ Returns a generator the will yield an infinite amout of training-batches ready to feed into the model"""
        return self._equalize_batch(self._batch(self._ngrammize(self._lookup_label(self._repeat(self.string_samples)))))

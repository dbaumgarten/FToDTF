""" Handles all evaluation relatet tasks, like printing the nearest neighbors of a word or visualizing the generated vectors"""

import tensorflow as tf


def print_similarity_check(similarity, words):
    """ Print similarity between given words
    :param similarity: A matrix of format len(words)xlen(words) containing the similarity between words
    :param list(str) words: Words to print the similarity for
    """
    for i, _ in enumerate(words):
        for j, _ in enumerate(words):
            print("Similarity between {} and {}: {:.2f}".format(
                words[i], words[j], similarity[i][j]))


class PrintValidationHook(tf.train.StepCounterHook):
    """ Implements a Hook that prints the resuls of the evaluation every x steps"""

    def __init__(self, every_n_steps, validationop, words):
        self.validationop = validationop
        self.every_n_steps = every_n_steps
        self.stepcounter = 0
        self.words = words
        super().__init__(self)

    def before_run(self, run_context):
        self.stepcounter += 1
        if self.stepcounter % self.every_n_steps == 0:
            self.stepcounter = 0
            return tf.train.SessionRunArgs([self.validationop])

    def after_run(self, run_context, run_values):
        results = run_values.results
        if results:
            print_similarity_check(results[0], self.words)

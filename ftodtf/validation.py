""" Handles all evaluation relatet tasks, like printing the nearest neighbors of a word or visualizing the generated vectors"""
import os
from tensorflow.contrib.tensorboard.plugins import projector

def printSimilarityCheck(similarityop,valid_examples,reverse_dictionary):
    """ Print the most similar words for given reference-words
    :param similarityop: The tensorflow operation returning the computed similarity
    :param list(int) valid_examples: Words to print he similarity for
    :param reverse_dictionary: A dict mapping ints (like in valid_examples) to their corresponding words
    """
    sim = similarityop.eval()
    print(sim)
    for i in range(len(valid_examples)):
        for j in range(len(valid_examples)):
            print("Similarity between {} and {}: {}".format(valid_examples[i],valid_examples[j],sim[i][j]))
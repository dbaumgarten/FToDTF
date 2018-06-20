""" Handles all evaluation relatet tasks, like printing the nearest neighbors of a word or visualizing the generated vectors"""
import os
from tensorflow.contrib.tensorboard.plugins import projector

def printSimilarityCheck(similarityop,words,reverse_dictionary):
    """ Print similarity between given words
    :param similarityop: The tensorflow operation returning the computed similarity
    :param list(str) words: Words to print he similarity for
    :param reverse_dictionary: A dict mapping ints (like in words) to their corresponding words
    """
    sim = similarityop.eval()
    for i in range(len(words)):
        for j in range(len(words)):
            print("Similarity between {} and {}: {:.2f}".format(words[i],words[j],sim[i][j]))
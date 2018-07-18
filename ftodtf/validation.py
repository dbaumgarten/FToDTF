""" Handles all evaluation relatet tasks, like printing the nearest neighbors of a word or visualizing the generated vectors"""


def print_similarity_check(similarityop, words, session):
    """ Print similarity between given words
    :param similarityop: The tensorflow operation returning the computed similarity
    :param list(str) words: Words to print he similarity for
    :param reverse_dictionary: A dict mapping ints (like in words) to their corresponding words
    """
    sim = session.run(similarityop)
    for i, _ in enumerate(words):
        for j, _ in enumerate(words):
            print("Similarity between {} and {}: {:.2f}".format(
                words[i], words[j], sim[i][j]))

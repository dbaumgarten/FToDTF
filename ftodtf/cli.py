""" This module handles parsing of cli-flags and then calls the needed function from the library"""
import argparse
import ftodtf.training

PARSER = argparse.ArgumentParser(description="Parse the hyperparameters for \
                                             the distributed FastText model.")
PARSER.add_argument('-l', '--log_dir', type=str,
                    help="The log directory for TensorBoard summaries.")

PARSER.add_argument('-s', '--steps', type=int,
                    help="The amount of training iterations.")

PARSER.add_argument('-v', '--vocabulary_size', type=int,
                    help="The number of unique words in the corpus.")

PARSER.add_argument('-b', '--batch_size', type=int,
                    help="The size of training samples to process in one go.")

PARSER.add_argument('-e', '--embedding_size', type=int,
                    help="Dimension of the embedding vector.")

PARSER.add_argument('-w', '--skip_window', type=int,
                    help="Words to consider left and right of the target word.")

PARSER.add_argument('-n', '--num_sampled', type=int,
                    help="Number of negative examples to sample.")

PARSER.add_argument('-c', '--corpus_path', type=str, required=True,
                    help='Path to the corpus/training data.')

PARSER.add_argument('-x', '--validation_words', type=str,
                    help='A comma seperated list of words to regularily compute similaritys of to check the training-progress',
                    required=True)


# TODO: Check for valid input of the user!
def check_valid_input(flag, value):
    """
    This function validates the input of the user.

    :param str flag: The name of the hyperparameter.
    :param value: The value of the hyperparameter provided by the user.
    :type value: int or str
    :return: True if the input is accurate.
    :raises: ValueError
    """
    if flag is not None and value is not None:
        if flag == 'log_dir':
            pass
        elif flag == 'steps':
            pass
        elif flag == 'vocabulary_size':
            pass
        elif flag == 'batch_size':
            pass
        elif flag == 'embedding_size':
            pass
        elif flag == 'skip_window' and value > 10:
            raise ValueError("The size of the window should be less than 10.")
        elif flag == 'num_skips':
            pass
        elif flag == 'num_sampled':
            pass
        elif flag == 'valid_size':
            pass
        elif flag == 'valid_window':
            pass
        elif flag == 'corpus_path':
            pass
        elif flag == 'validation_words':
            pass

        return True


def cli_main():
    """ Program entry point. """
    flags, _ = PARSER.parse_known_args()
    try:
        hyperparameter_values = {flag: val
                                 for (flag, val) in vars(flags).items()
                                 if check_valid_input(flag, val)}

    except ValueError as err:
        print(": ".join(["ERROR", err]))
    else:
        if "validation_words" in hyperparameter_values:
            hyperparameter_values["validation_words"] = hyperparameter_values["validation_words"].split(
                ",")

        ftodtf.training.train(**hyperparameter_values)


if __name__ == "__main__":
    cli_main()

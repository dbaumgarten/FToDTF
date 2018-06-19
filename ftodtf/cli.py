import argparse
import ftodtf.training

parser = argparse.ArgumentParser(description="Parse the hyperparameters for \
                                             the distributed FastText model.")
parser.add_argument('-l', '--log_dir', type=str,
                    help="The log directory for TensorBoard summaries.")

parser.add_argument('-s', '--steps', type=int,
                    help="The amount of training iterations.")

parser.add_argument('-v', '--vocabulary_size', type=int,
                    help="The number of unique words in the corpus.")

parser.add_argument('-b', '--batch_size', type=int,
                    help="The size of training samples to process in one go.")

parser.add_argument('-e', '--embedding_size', type=int,
                    help="Dimension of the embedding vector.")

parser.add_argument('-w', '--skip_window', type=int,
                    help="Words to consider left and right of the target word.")

parser.add_argument('-n', '--num_sampled', type=int,
                    help="Number of negative examples to sample.")

parser.add_argument('-V', '--valid_size', type=int,
                    help="Number of random words to use for validation.")

parser.add_argument('-W', '--valid_window', type=int,
                    help="Most frequent words to use for validation.")

parser.add_argument('-c', '--corpus_path', type=str, required=True,
                    help='Path to the corpus/training data.')

parser.add_argument('-x','--validation_words', type=str,
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
        if flag is 'log_dir':
            pass
        elif flag is 'steps':
            pass
        elif flag is 'vocabulary_size':
            pass
        elif flag is 'batch_size':
            pass
        elif flag is 'embedding_size':
            pass
        elif flag is 'skip_window' and value > 10:
            raise ValueError("The size of the window should be less than 10.")
        elif flag is 'num_skips':
            pass
        elif flag is 'num_sampled':
            pass
        elif flag is 'valid_size':
            pass
        elif flag is 'valid_window':
            pass
        elif flag is 'corpus_path':
            pass
        elif flag is 'validation_words':
            pass

        return True


def cli_main():
    """ Program entry point. """
    flags, _ = parser.parse_known_args()
    try:
        hyperparameter_values = {flag: val
                                 for (flag, val) in vars(flags).items()
                                 if check_valid_input(flag, val)}

    except ValueError as e:
        print(": ".join(["ERROR", e]))
    else:
        if "validation_words" in hyperparameter_values:
            hyperparameter_values["validation_words"] = hyperparameter_values["validation_words"].split(",")


        ftodtf.training.train(**hyperparameter_values)


if __name__ == "__main__":
    cli_main()

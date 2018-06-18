import argparse
import ftodtf.fasttext as fasttext

# TODO: Move this all to a config file!
parser = argparse.ArgumentParser(description="Parse the hyperparameters for \
                                             the distributed FastText model.")
parser.add_argument('--log_dir', type=str,
                    help="The log directory for TensorBoard summaries.")

parser.add_argument('--steps', type=int,
                    help="The amount of training iterations.")

parser.add_argument('--vocabulary_size', type=int,
                    help="The number of unique words in the corpus.")

parser.add_argument('--batch-size', type=int,
                    help="The size of training samples to process in one go.")

parser.add_argument('--embedding_size', type=int,
                    help="Dimension of the embedding vector.")

parser.add_argument('--skip_window', type=int,
                    help="Words to consider left and right of the target word.")

parser.add_argument('--num_skips', type=int,
                    help="Number of times to reuse an input to generate a label.")

parser.add_argument('--num_sampled', type=int,
                    help="Number of negative examples to sample.")

parser.add_argument('--valid_size', type=int,
                    help="Number of random words to use for validation.")

parser.add_argument('--valid_window', type=int,
                    help="")

parser.add_argument('--corpus_path', type=str,
                    help='Path to the corpus/training_data.')


# TODO: Check for valid input of the user!
def check_valid_input(flag, value):
    """
    This function validates the input of the user.
    :param flag: The name of the hyperparameter.
    :param value: The value of the hyperparameter provided by the user.
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
        fasttext.run(**hyperparameter_values)


if __name__ == "__main__":
    cli_main()

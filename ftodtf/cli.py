import argparse
import sys

import fasttext

def cli_main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        help='The log directory for TensorBoard summaries.')
    FLAGS, unparsed = parser.parse_known_args()

    #TODO: Add more arguments. See Issue #4


    set_vars = {k:v for (k,v) in vars(FLAGS).items() if v}
    fasttext.run(**set_vars)

if __name__ == "__main__":
    cli_main()
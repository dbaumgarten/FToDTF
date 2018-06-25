""" This module handles parsing of cli-flags and then calls the needed function from the library"""
import argparse
import sys
import ftodtf.training
from ftodtf.settings import FasttextSettings
import ftodtf.input

PREPROCESS_REQUIRED_PARAMETERS = ["corpus_path"]
TRAIN_REQUIRED_PARAMETERS = []
SETTINGS = FasttextSettings()
PARSER = argparse.ArgumentParser(
    description="Unsupervised training of word-vectors.")

SUBPARSER = PARSER.add_subparsers(dest="command")
PREPROCESS_PARSER = SUBPARSER.add_parser("preprocess")
TRAIN_PARSER = SUBPARSER.add_parser("train")


def add_arguments_to_parser(arglist, parser, required):
    """ Adds arguments (obtained from the settings-class) to an agrparse-parser

    :param list(str) arglist: A list of strings representing the names of the flags to add
    :param argparse.ArgumentParser parser: The parser to add the arguments to
    :param list(str) required: A list of argument-names that are required for the command
    """
    for parameter, default in filter(lambda x: x[0] in arglist, vars(SETTINGS).items()):
        parser.add_argument("--"+parameter, type=type(default),
                            help=SETTINGS.attribute_docstring(parameter), required=parameter in required, default=default)


add_arguments_to_parser(SETTINGS.preprocessing_settings(
), PREPROCESS_PARSER, PREPROCESS_REQUIRED_PARAMETERS)

add_arguments_to_parser(SETTINGS.training_settings(),
                        TRAIN_PARSER, TRAIN_REQUIRED_PARAMETERS)


def cli_main():
    """ Program entry point. """
    flags, unknown = PARSER.parse_known_args()
    if unknown:
        print(
            "Unknown flag '{}'. Run --help for a list of all possible flags".format(unknown[0]))
        sys.exit(1)
    # copy specified arguments over to the SETTINGS objct
    for k, v in vars(flags).items():
        SETTINGS.__dict__[k] = v
    try:
        SETTINGS.validate()
    except ValueError as err:
        print(": ".join(["ERROR", err]))
        sys.exit(1)

    if flags.command == "preprocess":
        ipp = ftodtf.input.InputProcessor(SETTINGS)
        ipp.preprocess()
        ipp.write_to_file(SETTINGS.batches_file)
    elif flags.command == "train":
        ftodtf.training.train(SETTINGS)
    else:
        PARSER.print_help()


if __name__ == "__main__":
    cli_main()

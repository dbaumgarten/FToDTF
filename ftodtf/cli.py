""" This module handles parsing of cli-flags and then calls the needed function from the library"""
import argparse
import sys
import ftodtf.training
from ftodtf.settings import FasttextSettings

SETTINGS = FasttextSettings()

PARSER = argparse.ArgumentParser(description="Set the hyperparameters for \
                                             the distributed FastText model.")


REQUIRED_PARAMETERS = ["corpus_path"]

for parameter, default in vars(SETTINGS).items():
    PARSER.add_argument("--"+parameter, type=type(default),
                        help=SETTINGS.attribute_docstring(parameter), required=parameter in REQUIRED_PARAMETERS, default=default)


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

    ftodtf.training.train(SETTINGS)


if __name__ == "__main__":
    cli_main()

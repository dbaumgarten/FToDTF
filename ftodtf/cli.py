""" This module handles parsing of cli-flags and then calls the needed function
from the library"""

import sys
import argparse
from multiprocessing import Process
from tqdm import tqdm
import ftodtf.model
import ftodtf.training
import ftodtf.input
from ftodtf.settings import FasttextSettings


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
                            help=SETTINGS.attribute_docstring(parameter),
                            required=parameter in required, default=default)


add_arguments_to_parser(SETTINGS.preprocessing_settings(),
                        PREPROCESS_PARSER,
                        PREPROCESS_REQUIRED_PARAMETERS)

add_arguments_to_parser(SETTINGS.training_settings(),
                        TRAIN_PARSER,
                        TRAIN_REQUIRED_PARAMETERS)


def spawn_progress_bar():
    """ This function will spawn a new process using multiprocessing module.

    :return: A child process.
    """
    p = Process(target=show_prog, args=(ftodtf.input.QUEUE, ))
    p.daemon = True
    return p


def show_prog(q):
    """ Show progressbar, converges against the next max progress_bar.n and
    finishes only when the function "write_batches_to_file" ends.

    :param q: Process which handles the progressbar.
    """
    proggress_bar = tqdm(total=100, desc="Segmen./Cleaning",
                         bar_format='{desc}:{percentage:3.0f}%|{bar}[{elapsed}]')
    n = 40
    j = 1
    while True:
        try:
            finished_function = q.get(timeout=1)
            if finished_function == "_process_text":
                proggress_bar.n = 66
                n, j = 10, 1
                proggress_bar.desc = "Writing Batches"
            elif finished_function == "write_batches_to_file":
                proggress_bar.n = 100
                proggress_bar.close()
                return 0
        except:
            if n <= 0:
                j *= 10
                n = j
            proggress_bar.update(1/j)
            n -= 1
            continue


def cli_main():
    """ Program entry point. """
    flags, unknown = PARSER.parse_known_args()
    if unknown:
        print(
            "Unknown flag '{}'. Run --help for a list of all possible "
            "flags".format(unknown[0]))
        sys.exit(1)
    # copy specified arguments over to the SETTINGS object
    for k, v in vars(flags).items():
        SETTINGS.__dict__[k] = v

    if flags.command == "preprocess":
        try:
            SETTINGS.validate_preprocess()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        else:
            p = spawn_progress_bar()
            p.start()
            ipp = ftodtf.input.InputProcessor(SETTINGS)
            ipp.preprocess()
            ftodtf.input.write_batches_to_file(ipp.batches(),
                                               SETTINGS.batches_file)
            p.join()
    elif flags.command == "train":
        try:
            SETTINGS.validate_train()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        else:
            ftodtf.training.train(SETTINGS)
    else:
        PARSER.print_help()


if __name__ == "__main__":
    try:
        cli_main()
    except KeyboardInterrupt as e:

        # Kill all subprocess
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.kill()

        print("Program interrupted!")
























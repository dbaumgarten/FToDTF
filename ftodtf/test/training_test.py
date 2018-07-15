# pylint: disable=missing-docstring
from tempfile import tempdir
from os.path import join
import os
import pytest
import ftodtf.training as ft
from ftodtf.settings import FasttextSettings
import ftodtf.input as inp


TESTDATA = """anarchism originated as a term of abuse first used against early
 working class radicals including the diggers of the english revolution and the
  sans culottes of the french revolution whilst the term is still used in a 
  pejorative way to describe any act that used violent means to destroy the 
  organization of society it has also been taken up as a positive label by 
  self defined anarchists the word anarchism is derived from the greek without 
  archons ruler chief king anarchism as a political"""


def setup_module():
    with open(join(tempdir, "TESTDATAfile"), "w") as file:
        file.write(TESTDATA)
    open(join(tempdir, "TESTBATCHfile"), "a").close()


def teardown_module():
    os.remove(join(tempdir, "TESTDATAfile"))
    os.remove(join(tempdir, "TESTBATCHfile"))


@pytest.mark.full
def test_fasttext():
    """run a single iteration just to check if it throws any errors.
    To skip this slow test run the test usering pytest -m 'not full' """

    seti = FasttextSettings()
    seti.corpus_path = join(tempdir, "TESTDATAfile")
    seti.steps = 1
    seti.vocabulary_size = 20
    seti.validation_words = "one,two,king,kingdom"
    seti.num_sampled = 3
    seti.batches_file = join(tempdir, "TESTBATCHfile")
    ipp = inp.InputProcessor(seti)
    ipp.preprocess()
    inp.write_batches_to_file(ipp.batches(10000), seti.batches_file)
    ft.train(seti)
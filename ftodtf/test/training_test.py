# pylint: disable=missing-docstring
from tempfile import tempdir
from os.path import join
import os
import pytest
import ftodtf.training as ft


TESTDATA = """anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organization of society it has also been taken up as a positive label by self defined anarchists the word anarchism is derived from the greek without archons ruler chief king anarchism as a political"""


def teardown_module():
    os.remove(join(tempdir, "TESTDATAfile"))


@pytest.mark.full
def test_fasttext():
    """run a single iteration just to check if it throws any errors. To skip this slow test run the test usering pytest -m 'not full' """
    with open(join(tempdir, "TESTDATAfile"), "w") as file:
        file.write(TESTDATA)
    ft.train(corpus_path=join(tempdir, "TESTDATAfile"), steps=1, vocabulary_size=20,
             num_sampled=3, validation_words=["one", "two", "king", "kingdom"])

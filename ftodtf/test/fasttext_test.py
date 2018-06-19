import ftodtf.fasttext as ft
import pytest
from tempfile import tempdir
from os.path import join
import os


testdata = """anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organization of society it has also been taken up as a positive label by self defined anarchists the word anarchism is derived from the greek without archons ruler chief king anarchism as a political"""

def teardown_module():
    os.remove(join(tempdir,"testdatafile"))

@pytest.mark.full
def test_fasttext():
    """run a single iteration just to check if it throws any errors. To skip this slow test run the test usering pytest -m 'not full' """
    with open(join(tempdir,"testdatafile"),"w") as f:
        f.write(testdata)
    ft.run(corpus_path=join(tempdir,"testdatafile"),steps=1,valid_window=10,valid_size=3,vocabulary_size=20,num_sampled=3)
# pylint: disable=missing-docstring,W0212
from tempfile import gettempdir
from os.path import join
import os

import ftodtf.input as inp


TESTFILECONTENT = """dies ist eine test datei
der text hier ist testtext
bla bla bla
"""

TESTFILENAME = join(gettempdir(), "ftodtftestfile")


def setup_module():
    with open(join(gettempdir(), TESTFILENAME), "w") as file:
        file.write(TESTFILECONTENT)


def teardown_module():
    os.remove(TESTFILENAME)


def test_generate_ngram_per_word():
    word = "diesisteintest"
    wanted = ["*di", "die", "ies", "esi", "sis", "ist", "ste", "tei",
              "ein", "int", "nte", "tes", "est", "st*", "*diesisteintest*"]
    ngrams = inp.generate_ngram_per_word(word, 3)
    assert len(ngrams) == len(wanted)
    for i, _ in enumerate(wanted):
        assert ngrams[i] == wanted[i]


def test_words_in_file():
    ipp = inp.InputProcessor(TESTFILENAME, 2, 128, 5, 3)
    expected = TESTFILECONTENT.replace("\n", " ").split()
    actual = ipp._words_in_file()
    i = 0
    for w in actual:
        assert w == expected[i]
        i += 1
    assert i == len(expected)


def test_preprocess():
    ipp = inp.InputProcessor(TESTFILENAME, 2, 128, 5, 3)
    ipp.preprocess()
    assert ipp.wordcount["bla"] == 3
    assert len(ipp.wordcount) == 10
    assert len(ipp.dict) == 5
    assert ipp.dict["bla"] == 0
    assert ipp.dict["ist"] == 1


def test_string_samples():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 128, 5, 3)
    ipp.preprocess()
    samples = ipp.string_samples()
    sample = samples.__next__()
    assert sample[0] == "dies"
    assert sample[1] == "ist"
    sample = samples.__next__()
    assert sample[0] == "ist"
    assert sample[1] == "dies" or sample[1] == "eine"


def test_string_samples2():
    ipp = inp.InputProcessor(TESTFILENAME, 10, 128, 5, 3)
    samples = list(ipp.string_samples())
    assert len(samples) == 13


def test_lookup_label():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 128, 5, 3)
    ipp.preprocess()
    testdata = [("bla", "ist")]
    labeled = list(ipp._lookup_label(testdata))
    assert labeled[0][0] == "bla"
    assert labeled[0][1] == 1


def test_repeat():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 128, 5, 3)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    repeated = ipp._repeat(generator_callable)
    for _ in range(20):
        repeated.__next__()
        # if we can get 20 elements from a two element generator repeat() seems to work


def test_batch():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 10, 5, 3)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    batchgen = ipp._batch(ipp._repeat(generator_callable))
    batch = batchgen.__next__()
    assert len(batch[0]) == 10
    assert len(batch[1]) == 10


def test_ngrammize():
    word = "diesisteintest"
    wantedngs = ["*di", "die", "ies", "esi", "sis", "ist", "ste", "tei",
                 "ein", "int", "nte", "tes", "est", "st*", "*diesisteintest*"]
    gener = (x for x in [(word, 1337)])
    ngramsize = 3
    ipp = inp.InputProcessor(TESTFILENAME, 1, 10, 5, ngramsize)
    ipp.preprocess()
    out = ipp._ngrammize(gener)
    output = out.__next__()
    assert len(output[0]) == len(wantedngs)
    assert output[1] == output[1]
    for i, _ in enumerate(wantedngs):
        assert output[0][i] == wantedngs[i]


def test_equalize_batch():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 128, 5, 3)
    ipp.preprocess()
    samples = ipp.batches().__next__()[0]
    for i in range(len(samples)-1):
        assert len(samples[i]) == len(samples[i+1])


def test_batches():
    ipp = inp.InputProcessor(TESTFILENAME, 1, 128, 5, 3)
    ipp.preprocess()
    batch = ipp.batches().__next__()
    print(batch)
    assert len(batch) == 2
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128
    assert isinstance(batch[0][0], list)
    assert isinstance(batch[1][0][0], int)

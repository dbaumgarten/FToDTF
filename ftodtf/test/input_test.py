# pylint: disable=missing-docstring,W0212
from tempfile import gettempdir
from os.path import join
import os

import ftodtf.input as inp
from ftodtf.settings import FasttextSettings
import fnvhash


TESTFILECONTENT = """dies ist eine test datei.
der text hier ist testtext.
bla bla bla.
"""

TESTFILENAME = join(gettempdir(), "ftodtftestfile")
TESTBATCHESFILENAME = join(gettempdir(), "ftodtftestbatchfile")

# Shared settings for all test-cases that are OK with the default values
SETTINGS = FasttextSettings()
SETTINGS.corpus_path = TESTFILENAME
SETTINGS.batches_file = TESTBATCHESFILENAME


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


def test_preprocess():
    seti = FasttextSettings()
    seti.corpus_path = TESTFILENAME
    seti.vocabulary_size = 5
    ipp = inp.InputProcessor(seti)
    ipp.preprocess()
    assert ipp.wordcount["bla"] == 3
    assert len(ipp.wordcount) == 10
    assert len(ipp.dict) == 5
    assert ipp.dict["bla"] == 0
    assert ipp.dict["ist"] == 1


def test_string_samples():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    samples = ipp.string_samples()
    sample = samples.__next__()
    assert sample[0] == "dies"
    while sample[0] != "ist":
        assert sample[1] in ["ist", "eine", "test", "datei", "der"]
        sample = samples.__next__()
    assert sample[0] == "ist"
    assert sample[1] in ["dies", "eine", "test", "datei", "der"]


def test_lookup_label():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    testdata = [("bla", "ist")]
    labeled = list(ipp._lookup_label(testdata))
    assert labeled[0][0] == "bla"
    assert labeled[0][1] == 1


def test_repeat():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    repeated = ipp._repeat(0, generator_callable)
    assert len(list(repeated)) == 2


def test_repeat2():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    repeated = ipp._repeat(2, generator_callable)
    assert len(list(repeated)) == 6


def test_repeat3():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    repeated = ipp._repeat(-1, generator_callable)
    for _ in range(20):
        repeated.__next__()


def test_batch():
    seti = FasttextSettings()
    seti.batch_size = 10
    seti.corpus_path = TESTFILENAME
    ipp = inp.InputProcessor(seti)
    ipp.preprocess()
    testdata = ["test", "generator"]
    # a callable that returns a finite generator

    def generator_callable():
        return (x for x in testdata)
    batchgen = ipp._batch(ipp._repeat(-1, generator_callable))
    batch = batchgen.__next__()
    assert len(batch[0]) == 10
    assert len(batch[1]) == 10


def test_ngrammize():
    word = "diesisteintest"
    wantedngs = ["*di", "die", "ies", "esi", "sis", "ist", "ste", "tei",
                 "ein", "int", "nte", "tes", "est", "st*", "*diesisteintest*"]
    gener = (x for x in [(word, 1337)])
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    out = ipp._ngrammize(gener)
    output = out.__next__()
    assert len(output[0]) == len(wantedngs)
    assert output[1] == output[1]
    for i, _ in enumerate(wantedngs):
        assert output[0][i] == wantedngs[i]


def test_equalize_batch():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    samples = ipp.batches(-1).__next__()[0]
    for i in range(len(samples)-1):
        assert len(samples[i]) == len(samples[i+1])


def test_hash_ngrams():
    testdata = [(["bla"], 1337)]
    wantedhash = (fnvhash.fnv1a_64("bla".encode("UTF-8")) %
                  (SETTINGS.num_buckets-1))+1
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    hashed = ipp._hash_ngrams(testdata)
    batch = hashed.__next__()
    assert batch[0][0] == wantedhash
    assert len(batch[0]) == 1
    assert batch[1] == 1337


def test_batches():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    batch = ipp.batches(-1).__next__()
    print(batch)
    assert len(batch) == 2
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128
    assert isinstance(batch[0][0], list)
    assert isinstance(batch[1][0], int)


def test_write_batches_to_file():
    ipp = inp.InputProcessor(SETTINGS)
    ipp.preprocess()
    batches = ipp.batches(100000)
    inp.write_batches_to_file(batches, SETTINGS.batches_file, 1)
    filesize = os.stat(SETTINGS.batches_file).st_size
    assert filesize > 100

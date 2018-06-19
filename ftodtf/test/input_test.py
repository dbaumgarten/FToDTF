from tempfile import gettempdir
from os.path import join
import os

import ftodtf.input as inp


testfilecontent = """dies ist eine test datei
der text hier ist testtext
bla bla bla
"""

testfilename = join(gettempdir(),"ftodtftestfile")

def setup_module():
    with open(join(gettempdir(),testfilename),"w") as f:
        f.write(testfilecontent)

def teardown_module():
    os.remove(testfilename)

def test_generate_ngram_per_word():
    word = "diesisteintest"
    wanted = ["*di","die","ies","esi","sis","ist","ste","tei","ein","int","nte","tes","est","st*","*diesisteintest*"]
    ngrams = inp.generate_ngram_per_word(word,3)
    assert len(ngrams) == len(wanted)
    for i in range(len(wanted)):
        assert ngrams[i] == wanted[i]

def test_words_in_file():
    ip = inp.InputProcessor(testfilename,2,128,5,3)
    expected = testfilecontent.replace("\n"," ").split()
    actual = ip._words_in_file()
    i = 0
    for w in actual:
        assert w == expected[i]
        i += 1
    assert i == len(expected)

def test_preprocess():
    ip = inp.InputProcessor(testfilename,2,128,5,3)
    ip.preprocess()
    assert ip.wordcount["bla"] == 3
    assert len(ip.wordcount) == 10
    assert len(ip.dict) == 5
    assert ip.dict["bla"] == 0
    assert ip.dict["ist"] == 1
    assert len(ip.dict) == len(ip.reversed_dict)
    assert ip.reversed_dict[0] == "bla"

def test_string_samples():
    ip = inp.InputProcessor(testfilename,1,128,5,3)
    ip.preprocess()
    samples = ip.string_samples()
    sample = samples.__next__()
    assert sample[0] == "dies"
    assert sample[1] == "ist"
    sample = samples.__next__()
    assert sample[0] == "ist"
    assert sample[1] == "dies" or sample[1] == "eine"

def test_string_samples2():
    ip = inp.InputProcessor(testfilename,10,128,5,3)
    samples = list(ip.string_samples())
    assert len(samples) == 13

def test_lookup_label():
    ip = inp.InputProcessor(testfilename,1,128,5,3)
    ip.preprocess()
    testdata = [("bla","ist")]
    labeled = list(ip._lookup_label(testdata))
    assert labeled[0][0] == "bla"
    assert labeled[0][1] == 1

def test_repeat():
    ip = inp.InputProcessor(testfilename,1,128,5,3)
    ip.preprocess()
    testdata = ["test","generator"]
    # a callable that returns a finite generator
    def generator_callable():
        return (x for x in testdata)
    repeated = ip._repeat(generator_callable)
    for _ in range(20):
        repeated.__next__()
        # if we can get 20 elements from a two element generator repeat() seems to work

def test_batch():
    ip = inp.InputProcessor(testfilename,1,10,5,3)
    ip.preprocess()
    testdata = ["test","generator"]
    # a callable that returns a finite generator
    def generator_callable():
        return (x for x in testdata)
    batchgen = ip._batch(ip._repeat(generator_callable))
    batch = batchgen.__next__()
    assert len(batch[0]) == 10
    assert len(batch[1]) == 10

def test_ngrammize():
    word = "diesisteintest"
    wantedngs = ["*di","die","ies","esi","sis","ist","ste","tei","ein","int","nte","tes","est","st*","*diesisteintest*"]
    gener = (x for x in [(word,1337)])
    ngramsize = 3
    ip = inp.InputProcessor(testfilename,1,10,5,ngramsize)
    ip.preprocess()
    out = ip._ngrammize(gener)
    output = out.__next__()
    assert len(output[0]) == len(wantedngs)
    assert output[1] == output[1]
    for i in range(len(wantedngs)):
        assert output[0][i] == wantedngs[i]

def test_equalize_batch():
    ip = inp.InputProcessor(testfilename,1,128,5,3)
    ip.preprocess()
    samples =  ip.batches().__next__()[0]
    for i in range(len(samples)-1):
        assert len(samples[i]) == len(samples[i+1])


def test_batches():
    ip = inp.InputProcessor(testfilename,1,128,5,3)
    ip.preprocess()
    batch = ip.batches().__next__()
    print(batch)
    assert len(batch) == 2
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128
    assert isinstance(batch[0][0],list)
    assert isinstance(batch[1][0][0],int)

setup_module()
test_batches()


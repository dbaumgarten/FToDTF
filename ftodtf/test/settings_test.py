# pylint: disable=missing-docstring
import pytest
import ftodtf.settings


def test_class_FasttextSettings():
    f = ftodtf.settings.FasttextSettings()
    assert f is not None


def test_preprocessing_settings():
    pre_seti = {"corpus_path", "batches_file", "vocabulary_size", "batch_size",
            "skip_window", "ngram_size", "num_buckets", "rejection_threshold",
            "profile"}

    assert len(pre_seti) == len(ftodtf.settings.FasttextSettings.preprocessing_settings())
    assert pre_seti == set(ftodtf.settings.FasttextSettings.preprocessing_settings())


def test_training_settings():
    train_seti = {"batches_file", "log_dir", "steps", "vocabulary_size",
            "batch_size", "embedding_size", "num_sampled", "num_buckets",
            "validation_words", "profile", "learnrate", "save_mode"}
    assert len(train_seti) == len(ftodtf.settings.FasttextSettings.training_settings())
    assert train_seti == set(ftodtf.settings.FasttextSettings.training_settings())



def test_validation_word_list():
    seti = ftodtf.settings.FasttextSettings()
    seti.validation_words = "i,am,groot"
    vwli = seti.validation_words_list
    wanted = ["i", "am", "groot"]
    for i, v in enumerate(wanted):
        assert v == vwli[i]


def test_attribute_docstring():
    seti = ftodtf.settings.FasttextSettings()
    assert seti.attribute_docstring(
        "corpus_path", False) == "Path to the file containing text for training the model."
    assert seti.attribute_docstring(
        "num_sampled", True) == "Number of negative examples to sample when computing the nce_loss. Default: 5"


def test_validate_preprocess():
    seti = ftodtf.settings.FasttextSettings()
    seti.batches_size = 10000
    with pytest.raises(Exception):
        seti.validate_preprocess()

    seti.embedding_size = 10000
    with pytest.raises(Exception):
        seti.validate_train()

def test_corpus_path():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_corpus_path("/fake/folder")


@pytest.mark.parametrize("test_input", [-1, 10251099])
def test_check_vocabulary_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_vocabulary_size(test_input)

@pytest.mark.parametrize("test_input", [-1, 2])
def test_check_rejection_threshold(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_rejection_threshold(test_input)


@pytest.mark.parametrize("test_input", [31, 1025])
def test_check_batch_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_batch_size(test_input)


@pytest.mark.parametrize("test_input", [0,6])
def test_check_skip_window(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_skip_window(test_input)


@pytest.mark.parametrize("test_input", [2,7])
def test_check_ngram_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_ngram_size(test_input)


def test_num_buckets():
    with pytest.raises(ValueError):
        ftodtf.settings.check_num_buckets(-1)


def test_batches_file():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_batches_file('/fake/file/')


def test_check_log_dir():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_log_dir("/fake/log/folder")


def test_check_steps():
    with pytest.raises(ValueError):
        ftodtf.settings.check_steps(-1)

@pytest.mark.parametrize("test_input", [49, 1001])
def test_check_embedding_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_embedding_size(test_input)


@pytest.mark.parametrize("test_input", [-1, 11])
def test_check_num_sampled(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_num_sampled(test_input)

@pytest.mark.parametrize("test_input", [-1,2])
def test_check_learn_rate(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_learn_rate(test_input)








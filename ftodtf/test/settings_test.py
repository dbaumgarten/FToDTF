# pylint: disable=missing-docstring
import ftodtf.settings


def test_validation_word_list():
    seti = ftodtf.settings.FasttextSettings()
    seti.validation_words = "i,am,groot"
    vwli = seti.validation_words_list
    wanted = ["i", "am", "groot"]
    for i, v in enumerate(wanted):
        assert v == vwli[i]


def test_validate():
    # Not yet implemeted
    pass


def test_attribute_docstring():
    seti = ftodtf.settings.FasttextSettings()
    assert seti.attribute_docstring(
        "corpus_path",) == "Path to the file containing text for training the model."
    assert seti.attribute_docstring(
        "num_sampled",) == "Number of negative examples to sample when computing the nce_loss."

import ftodtf.fasttext as ft
import pytest

@pytest.mark.full
def test_fasttext():
    """run a single iteration just to check if it throws any errors.
    to skip this slow test run the test usering pytest -m 'not full' """
    # Currently it will download the default dataset, which can take a big. One the dataset can be specified we should switch to a dummy-dataset for this
    # also the evaluation at the  end of training slows things down massively
    ft.run(steps=1)
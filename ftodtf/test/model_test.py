"""ATTENTION!: When running pytest the  command "LD_LIBRARY_PATH=/usr/lib/nvidia-396 pytest" must be used.
I have no idea why, just do it..."""
import ftodtf.model as model

def test_model():
    #We have no idea, if the graph is correct, but at least the function ran without errors and returned something
    m = model.Model(128,300,50000,[1,2,3,4],16)
    assert m
    assert m.loss is not None
    assert m.merged is not None
    assert m.optimizer is not None
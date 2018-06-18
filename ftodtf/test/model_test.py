import ftodtf.model as model

def test_model():
    #We have no idea, if the graph is correct, but at least the function ran without errors and returned something

    def batches():
        return [(x,[x]) for x in range(1,128)]


    m = model.Model(batches,128,300,50000,[1,2,3,4],16)
    assert m
    assert m.loss is not None
    assert m.merged is not None
    assert m.optimizer is not None
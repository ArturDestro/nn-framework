import numpy as np
from framework.model import *
from framework.layers import *

def test_dropout_train_eval_modes():
    d = Dropout(0.5)
    x = np.ones((10,4))
    d.training = True
    y1 = d.forward_pass(x)
    assert not np.allclose(y1, x)  # deve desligar alguns
    d.training = False
    y2 = d.forward_pass(x)
    assert np.allclose(y2, x)

def test_model_no_layers_evaluate_empty():
    m = Model()
    X = np.zeros((0, 3))
    y = np.zeros((0, 1))
    res = m.evaluate(X, y, metrics=[], loss=None)
    # deve lidar com n==0 sem crash; checar que não lança
    assert isinstance(res, dict)

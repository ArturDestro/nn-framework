import numpy as np
from framework.model import *
from framework.layers import *

def test_mse_loss_backward():
    loss = MSELoss()
    y_pred = np.array([[1.0], [2.0]])
    y_true = np.array([[0.0], [2.0]])
    l = loss.forward_pass(y_pred, y_true)
    grad = loss.backward_pass()
    assert grad.shape == y_pred.shape

def test_crossentropy_and_accuracy():
    loss = CrossEntropyLoss()
    y_logits = np.array([[2.0, 1.0], [0.1, 3.0]])
    y_true = np.array([[1,0],[0,1]])
    l = loss.forward_pass(y_logits, y_true)
    grad = loss.backward_pass()
    assert grad.shape == y_logits.shape

    acc = Accuracy()
    acc.reset()
    acc.update(y_logits, y_true)
    assert isinstance(acc.compute(), float)

import numpy as np
from framework.model import *
from framework.layers import *

def test_dense_forward_backward_shapes():
    rng = np.random.RandomState(0)
    init = HeNormal()
    dense = Dense(entradas=3, saidas=2, initializer=init)
    x = rng.randn(5, 3)
    y = dense.forward_pass(x)
    assert y.shape == (5, 2)
    g = np.ones_like(y)
    dx = dense.backward_pass(g)
    assert dx.shape == x.shape
    assert hasattr(dense, "dw") and hasattr(dense, "db")

def test_relu_behavior():
    relu = ReLU()
    x = np.array([[-1.0, 0.0, 2.0]])
    y = relu.forward_pass(x)
    assert (y >= 0).all()
    g = np.ones_like(y)
    dx = relu.backward_pass(g)
    assert dx[0,0] == 0.0 and dx[0,2] == 1.0

def test_flatten_roundtrip():
    fl = Flatten()
    x = np.zeros((4,2,3,1))
    y = fl.forward_pass(x)
    assert y.shape[0] == 4
    xr = fl.backward_pass(y)
    assert xr.shape == x.shape

def test_maxpool_forward_backward():
    mp = MaxPool(size=(2,2), stride=2)
    x = np.array([[[[1.0],[2.0]],[[3.0],[0.5]]]])  # shape (1,2,2,1)
    y = mp.forward_pass(x)
    assert y.shape == (1,1,1,1)
    g = np.ones_like(y)
    dx = mp.backward_pass(g)
    assert dx.shape == x.shape

import numpy as np
from framework.model import *
from framework.layers import *

def test_training_decreases_loss():
    rng = np.random.RandomState(0)
    # dataset: y = 2*x + 1
    X = rng.randn(100,1)
    y = 2.0 * X + 1.0
    m = Model()
    m.add(Dense(entradas=1, saidas=1, initializer=HeNormal()))
    loss = MSELoss()
    opt = SGD(lr=0.1)
    hist = m.fit(X, y, loss=loss, optimizer=opt, epochs=10, batch_size=16, shuffle=True, verbose=False)
    assert "loss" in hist and hist["loss"][-1] <= hist["loss"][0]

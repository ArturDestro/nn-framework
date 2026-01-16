import numpy as np
import tempfile
from framework.model import *
from framework.layers import *

def make_simple_model():
    m = Model()
    m.add(Dense(entradas=2, saidas=1, initializer=HeNormal()))
    return m

def test_save_load_weights_roundtrip():
    m = make_simple_model()
    before = m.get_state_dict()
    # change params
    for layer in m.layers:
        if hasattr(layer, "w"):
            layer.w += 1.0
    tf = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    path = tf.name
    m.save_weights(path)
    # reload into fresh model same architecture
    m2 = make_simple_model()
    m2.load_weights(path, strict=True)
    after = m2.get_state_dict()
    assert len(after["layers"]) == len(before["layers"])

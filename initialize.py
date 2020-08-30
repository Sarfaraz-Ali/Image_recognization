
import numpy as np 

def initialize(dim):
    w = np.zeros(dim).reshape(dim,1)
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
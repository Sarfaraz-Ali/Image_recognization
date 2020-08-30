
import numpy as np 
from sigmoid import sigmoid

def propagate(w, b, X, Y):
    
    
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)                                
    cost = -np.sum((Y*np.log(A))+((1-Y)*np.log(1-A)))/m         
    
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum((A-Y)/m)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
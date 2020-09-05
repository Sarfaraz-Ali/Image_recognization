
import numpy as np 
from sigmoid import sigmoid 

def predict(w, b, X):
    
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    Y_prediction = (A>=0.5)*1
        
    return Y_prediction
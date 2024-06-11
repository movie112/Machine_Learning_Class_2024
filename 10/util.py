import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def compute_activation(z):
    # ==================================================
    # complete the code
    # ==================================================
    z = 1 / (1 + np.exp(-z))
    return z
    

def compute_prediction(x, y, theta):
    # ==================================================
    # complete the code
    # ==================================================
    z = np.dot(np.c_[np.ones(x.shape[0]), x, y], theta)
    pred = compute_activation(z)
    pred = (pred >= 0.5).astype(int)
    return pred 


def compute_true_positive(label, pred):
    # ==================================================
    # complete the code
    # ==================================================
    tp = np.sum((label == 1) & (pred == 1))
    return tp 
   
    
def compute_false_positive(label, pred):
    # ==================================================
    # complete the code
    # ==================================================
    fp = np.sum((label == 0) & (pred == 1))
    return fp 

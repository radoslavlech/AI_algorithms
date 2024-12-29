import numpy as np

def heaviside(s):
    if s >= 0:
        return 1
    else:
        return 0

def sigmoid(s,beta):
    return 1/(1+np.e**(-beta*s))




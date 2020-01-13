import numpy as np
import pandas as pd

def normalize(matrix,flat):
    mean = np.mean(flat)
    sigma = np.std(flat)
    mat = matrix-mean
    mat /= sigma
    return mat,mean,sigma
    
def denormalize(matrix,mean,sigma):
    matrix*=sigma
    matrix+=mean
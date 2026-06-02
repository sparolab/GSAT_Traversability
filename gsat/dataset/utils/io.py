import numpy as np

def load_bin(filename):
    points = np.fromfile(filename, dtype=np.float32)
    points = points.reshape(-1, 4)
    
    
    return points
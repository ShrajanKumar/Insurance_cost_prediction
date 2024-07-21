import numpy as np
import pandas as pd

def log1p_transform(X):
     return np.log1p(X.astype(float)).reshape(-1, 1)
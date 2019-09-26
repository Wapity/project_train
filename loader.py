import pandas as pd
from sklearn.utils import shuffle

import numpy as np

def df_to_X_y(df, index_start_x, index_y, index_end_x = None):

    if index_end_x == None:
        X = np.array(df.iloc[:, index_start_x:])
    else :
        X = np.array(df.iloc[:, index_start_x:index_end_x])

    y = np.array(df.iloc[:,index_y])

    return X, y

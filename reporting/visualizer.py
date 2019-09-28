
import matplotlib.pyplot as plt
import pandas as pd
from paths import PATH_PRINCETON_DATASET
import numpy as np
from math import *

from loader import df_to_X_y  # packages a importer aussi
from benchmark_models import benchmark_models


catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

X, y = df_to_X_y(catalog, 1, 0)

#image = np.array(X[0].reshape(50, 50)) # pas générique, dimmensions telles que sqrt(p) soit entier
#image = np.expand_dims(image , axis=2)

def visualizer_image(image):
    plt.imshow(image)
    plt.show()

def visualizer_results_images(X_test,y_pred):
    n,p = int(X.shape) #possible mettre dans utils
    for i in range(n+1):
        X[i]= np.array(X[i].reshape(sqrt(p),sqrt(p)))
        fig = plt.figure()
        imgplot = plt.imshow(X[i])
        plt.title("Predicted %d " %y_pred)

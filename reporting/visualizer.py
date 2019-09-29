
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt

def visualizer_images(X, row_number):
    converted_image = X[row_number].reshape(50,50)
    plt.imshow(converted_image)
    plt.show()

def visualizer_results_images(X,y, y_pred, row_number):
    converted_image = X[row_number].reshape(50,50)
    plt.title("Real value {0} / Predicted value {1} ".format(y[row_number], y_pred[row_number]))
    plt.imshow(converted_image)
    plt.show()

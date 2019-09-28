
import matplotlib.pyplot as plt
import pandas as pd
from paths import PATH_PRINCETON_DATASET
import numpy as np


from loader import df_to_X_y  # packages a importer aussi


catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

X, y = df_to_X_y(catalog, 1, 0)
image = np.array(X[0].reshape(50, 50))
#image = np.expand_dims(image , axis=2)
print(image.shape)


def visualizer_image(image):
    plt.imshow(image)
    plt.show()

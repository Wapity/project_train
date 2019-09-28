
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt


def visualizer_image(image):
    plt.imshow(image)
    plt.show()

def visualizer_results_images(X_test,y_pred):
    ma_list = np.arrays([])
    for i in range(10):
        print(X_test.shape)
        ma_list.append(np.array(X_test[i].reshape(50,50)))
        plt.figure()
        imgplot = plt.imshow(ma_list)
        plt.title("Predicted %d " %y_pred)



if __name__ == '__main__':

    def df_to_X_y(df, index_start_x, index_y, index_end_x=None):

        if index_end_x == None:
            X = np.array(df.iloc[:, index_start_x:])
        else:
            X = np.array(df.iloc[:, index_start_x:index_end_x])

        y = np.array(df.iloc[:, index_y])

        return X, y


    PATH_PRINCETON_DATASET = '/Users/alex/project/project_train/data/princeton-dataset_2019-09-02/'

    catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

    X, y = df_to_X_y(catalog, 1, 0)

    visualizer_results_images(X,y)

import pandas as pd

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import writer, reader

import matplotlib.pyplot as plt

from loader import df_to_X_y
from benchmark_models import benchmark_models
from reporting.visualizer import visualizer_images, visualizer_results_images
from lib.lib import listener

from paths import PATH_PRINCETON_DATASET

import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples.")


if __name__ == '__main__':

    '''catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

    X, y = df_to_X_y(catalog, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True)'''

    # input_model, cross_validation = listener()

    '''
    y_pred, best_model, score_max  = benchmark_models(
        X_train, X_test, y_train, y_test, 'svm', tuning=False,full_reporting=False)

    writer(best_model, 'best_model')

    print(best_model)
    print(score_max)

    '''

    best_model = reader('best_model')

    abcd = pd.read_csv(PATH_PRINCETON_DATASET + '/tests/abcdefghijklmnopqrstuvwxyz_0.csv')

    y_pred = best_model.predict(abcd.iloc[:, 3:])

    abcd[y_pred==False].plot.scatter('x', 'y')
    plt.show()
    #visualizer_results_images(X_test, y_pred)

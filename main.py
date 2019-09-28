import pandas as pd

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from loader import df_to_X_y
from benchmark_models import benchmark_models
from reporting.visualizer import visualizer_results_images
from lib.lib import listener

from paths import PATH_PRINCETON_DATASET

import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples.")



if __name__ == '__main__':

    catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

    X, y = df_to_X_y(catalog, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True)

    input_model, cross_validation = listener()
    #best_score, best_model =
    y_pred, best_model, score_max  = benchmark_models(
        X_train, X_test, y_train, y_test, input_model, tuning=cross_validation,full_reporting=False)

    print(best_model)
    print(score_max)

    #visualizer_results_images(X_test, y_pred)

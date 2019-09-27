import pandas as pd

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from loader import df_to_X_y
from benchmark_models import benchmark_models
from paths import PATH_PRINCETON_DATASET


if __name__ == '__main__':

    catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows=500)

    X, y = df_to_X_y(catalog, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True)

    import sys
    param_tuning = sys.argv[1]


    if param_tuning == 'true':
        tuning=True
        benchmark_models(
            X_train, X_test, y_train, y_test, 'svm', tuning=tuning)

    else :
        tuning=False
        benchmark_models(
            X_train, X_test, y_train, y_test, 'svm', tuning=tuning)
        print(best_score, best_model)

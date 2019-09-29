
from utils import value_index_max
import matplotlib.pyplot as plt
from termcolor import colored

from models import house_models, house_models_cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from reporting.benchmark_cv_full_reporting import benchmark_cv_full_reporting


def benchmark_models(X_train, X_test, y_train, y_test, famille_de_model, tuning=False, nb_cv=5, full_reporting=False):
    if not tuning:
        scores = []
        model_list = house_models[famille_de_model]
        for model in model_list:
            model['model'].fit(X_train, y_train)
            scores.append(model['model'].score(X_test, y_test))

        index_best_model, score_max = value_index_max(scores)

        best_model = model_list[index_best_model]
        y_pred = best_model['model'].predict(X_test)

        return y_pred, best_model['model'], score_max


    model = house_models_cross_validation[famille_de_model]['model']
    tuned_parameters = house_models_cross_validation[famille_de_model]['param_cross_validation']

    if not full_reporting:

        # verbose = 2 to display the parameters used
        clf = GridSearchCV(model, tuned_parameters,
                           cv=nb_cv, scoring='precision_macro')
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)

        best_model = clf.best_estimator_
        best_score = clf.best_score_

        return y_pred, best_model, best_score

    return benchmark_cv_full_reporting(X_train,y_train,X_test, y_test, model,tuned_parameters, nb_cv=5)

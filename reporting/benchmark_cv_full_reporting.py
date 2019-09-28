from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from termcolor import colored

from models import house_models, house_models_cross_validation


def benchmark_cv_full_reporting(X_train, y_train, X_test, y_test, model, tuned_parameters, nb_cv):

    print(model, tuned_parameters)

    print("# Tuning hyper-parameters for %s" % 'precision')
    print()

    clf = GridSearchCV(model, tuned_parameters,
                       cv=nb_cv, scoring='%s_macro' % 'precision', verbose=2)  # verbose = 2 to display the parameters used
    clf.fit(X_train, y_train)

    print(colored("Best parameters set found on development set:", 'red'))
    print()
    print(colored(clf.best_params_, 'red'))
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    best_model = clf.best_estimator_
    best_score = clf.best_score_
    return y_pred, best_model, best_score

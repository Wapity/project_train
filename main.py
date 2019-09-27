import pandas as pd

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from loader import df_to_X_y
from benchmark_models import benchmark_models
from models import house_models
from paths import PATH_PRINCETON_DATASET



if __name__ == '__main__':

    catalog = pd.read_csv(PATH_PRINCETON_DATASET + 'catalog.csv', nrows = 500)

    X, y = df_to_X_y(catalog, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.10, random_state=42, shuffle = True)

    best_score, best_model = benchmark_models(X_train, X_test, y_train, y_test, house_models['svm'], tuning = False)
    print(best_score, best_model)













    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    print("# Tuning hyper-parameters for %s" % 'precision')
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % 'precision')
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
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

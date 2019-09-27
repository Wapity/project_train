
from utils.list import value_index_max

from models import house_models, house_models_cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def benchmark_models(X_train, X_test, y_train, y_test, famille_de_model, tuning=False, nb_cv=5):
    if not tuning:
        scores = []
        model_list = house_models[famille_de_model]
        for model in model_list:
            model['model'].fit(X_train, y_train)
            scores.append(model['model'].score(X_test, y_test))

        index_best_model, score_max = value_index_max(scores)
        return score_max, model_list[index_best_model]

    else:
        # balancer toute la partie grid search
        # Set the parameters by cross-validation
        model = house_models_cross_validation[famille_de_model]['model']
        tuned_parameters = house_models_cross_validation[famille_de_model]['param_cross_validation']
        print(model)

        print("# Tuning hyper-parameters for %s" % 'precision')
        print()

        clf = GridSearchCV(model, tuned_parameters, cv = nb_cv, scoring='precision_macro')
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

from sklearn.svm import SVC, LinearSVC

C = 1.0  # SVM regularization parameter

house_models = {
    'svm': [{
        'model': SVC(kernel='linear', C=C),
        'param_cross_validation': None
    },
        {
        'model': LinearSVC(C=C, max_iter=10000),
        'param_cross_validation': None
    },
        {
        'model': SVC(kernel='rbf', gamma=0.7, C=C),
        'param_cross_validation': None
    },
        {
        'model': SVC(kernel='poly', degree=3, gamma='auto', C=C),
        'param_cross_validation': None
    }
    ]
}


house_models_cross_validation = {
    'svm': {
        'model': SVC(),
        'param_cross_validation': [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    }
}

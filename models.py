from sklearn import svm

C = 1.0  # SVM regularization parameter
'''clf_linear = svm.SVC(kernel = 'linear', C = C)
clf_kernel_linear = svm.LinearSVC(C = C, max_iter = 10000)
clf_kernel_rbf = svm.SVC(kernel = 'rbf', gamma = 0.7, C = C)
clf_poly = svm.SVC(kernel = 'poly', degree = 3, gamma = 'auto', C = C)'''

house_models = {
    'svm': [{
        'model': svm.SVC(kernel='linear', C=C),
        'param_cross_validation': None
    },
        {
        'model': svm.LinearSVC(C=C, max_iter=10000),
        'param_cross_validation': None
    },
        {
        'model': svm.SVC(kernel='rbf', gamma=0.7, C=C),
        'param_cross_validation': None
    },
        {
        'model': svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
        'param_cross_validation': None
    }
    ]
}

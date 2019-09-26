
from models import house_models


from utils.list import value_index_max

def benchmark_models(X_train, X_test, y_train, y_test, models):
    scores = []
    for model in models :
        model.fit(X_train,y_train)
        scores.append(model.score(X_test, y_test))

    index_best_model, score_max = value_index_max(scores)
    return score_max, models[index_best_model]

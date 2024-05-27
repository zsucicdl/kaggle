from sklearn.metrics import cohen_kappa_score


def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results
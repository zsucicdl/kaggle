from sklearn.metrics import cohen_kappa_score


def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0, 5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results
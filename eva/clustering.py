import numpy as np
from sklearn import metrics


def purity_score(y_true, y_pred):
    """Compute purity score for clustering evaluation"""
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_metrics(labels, preds):
    """Calculate clustering evaluation metrics"""
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results


def _clustering(theta, labels):
    """Perform clustering evaluation using theta (document-topic distributions)"""
    preds = np.argmax(theta, axis=1)
    return clustering_metrics(labels, preds)


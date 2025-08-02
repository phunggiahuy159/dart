import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score


def _cls(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale'):
    """Perform classification evaluation using theta representations"""
    if classifier == 'SVM':
        clf = SVC(gamma=gamma)
    else:
        raise NotImplementedError

    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)
    results = {
        'acc': accuracy_score(test_labels, preds),
        'macro-F1': f1_score(test_labels, preds, average='macro')
    }
    return results
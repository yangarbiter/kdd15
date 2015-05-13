
import feature, proc
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc

def main():
    with open(proc.DATA_PATH_PREFIX+'data.pkl', 'r') as f:
        data = cPickle.load(f)

    X = feature.build_feature(data)

    skf = cross_validation.StratifiedKFold(y, n_folds=3)
    for train_idx, test_idx in skf:
        clf = LinearSVC()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        auc(X_test, y_test)


if __name__ == "__main__":
    main()

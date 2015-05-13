
import feature, proc, cPickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression

def main():
    with open(proc.DATA_PATH_PREFIX+'data.pkl', 'r') as f:
        data = cPickle.load(f)
    del data[139669] #data without log

    X = feature.build_feature(data)
    y = [ proc.loadans()[i] for i in data.keys() ]

    print np.shape(X), np.shape(y)
    skf = StratifiedKFold(y, n_folds=3)
    for train_idx, test_idx in skf:
        clf = LogisticRegression(
                C=1.0
            )
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        print auc(X_test, y_test)


if __name__ == "__main__":
    main()

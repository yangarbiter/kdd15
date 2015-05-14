
import feature, proc, cPickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

def main():
    print "============ loading data ==============="
    #with open(proc.DATA_PATH_PREFIX+'data.pkl', 'r') as f:
    #    data = cPickle.load(f)
    #del data[139669] #data without log
    #ans = proc.loadans()

    #print "============ building feature ==============="
    #X = feature.build_feature(data)
    #y = [ ans[i] for i in data.keys() ]

    #with open('xy.pkl', 'w') as f:
    #    cPickle.dump((X, y), f)

    with open('xy.pkl', 'r') as f:
        X, y = cPickle.load(f)

    X = np.array(X)
    y = np.array(y)
    print np.shape(X), np.shape(y)

    #X = X[:, :11]

    for i in range(np.shape(X)[1]):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))

    print "============ trainging ==============="
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        print c, "=========================="
        skf = StratifiedKFold(y, n_folds=5, shuffle=True)
        for train_idx, test_idx in skf:
            clf = LogisticRegression(
                        C=c
                )
            #clf = LinearSVC(
            #            C=c
            #    )
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            print clf.score(X_test, y_test)
            #fpr, tpr, thresholds = roc_curve(y_test, clf.predict(X_test), pos_label=1)
            #print auc(fpr, tpr)
            print roc_auc_score(y_test, clf.predict(X_test))
            print "------------------------"


if __name__ == "__main__":
    main()


import feature, proc, cPickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def main():
    print "============ loading data ==============="
    with open('xy.pkl', 'r') as f:
        X, y = cPickle.load(f)

    X = np.array(X)
    y = np.array(y)
    print np.shape(X), np.shape(y)

#normalize to [0, 1]
    for i in range(np.shape(X)[1]):
        if np.sum((X[:, i]!=0)) == 0: continue
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))

    print "============ trainging ==============="

    """For generating output"""
    #with open('./testx.pkl', 'r') as f:
    #    idxs, testX = cPickle.load(f)

    #for i in range(np.shape(testX)[1]):
    #    if np.sum((testX[:, i]!=0)) == 0: continue
    #    testX[:, i] = (testX[:, i] - np.min(testX[:, i])) / (np.max(testX[:, i]) - np.min(testX[:, i]))
    #clf = LogisticRegression(
    #        penalty="l2",
    #        C=1000,
    #        class_weight='auto'
    #    )
    #clf.fit(X, y)
    #print clf.predict_proba(X)[:, 1]
    #print testX
    #proc.outputans(idxs, clf.predict_proba(testX)[:, 1])
    #exit()
    """"""

    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        print c, "=========================="
        skf = StratifiedKFold(y, n_folds=5, shuffle=True)
        aucs = []
        for train_idx, test_idx in skf:
            #clf = RandomForestClassifier(
            #        n_estimators = 2000,
            #        class_weight = "auto",
            #        max_depth = 10,
            #        n_jobs=-1
            #    )

            clf = LogisticRegression(
                    penalty="l2",
                    C=c,
                    class_weight='auto'
                )

            #clf = LinearSVC(
            #            C=c
            #    )
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            #print clf.score(X_test, y_test)
            #fpr, tpr, thresholds = roc_curve(y_test, clf.predict(X_test), pos_label=1)
            #print auc(fpr, tpr)
            print clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        print np.mean(aucs)







if __name__ == "__main__":
    main()

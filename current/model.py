
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pandas.io.parsers import read_csv

def evaluate(X_train, y_train, X_test, y_test, n_round):
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dval = xgb.DMatrix(X_test, label = y_test)
    param = {'bst:max_depth': 6,
             'bst:eta': 0.05,
             'lambda': 0.01,
             'gamma': 0,
             'objective':'binary:logistic',
             'eval_metric': 'auc',
             'silent': 1}
    plst = param.items()
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    num_round = n_round
    bst = xgb.train(plst, dtrain, num_round, evallist)
    ypred = bst.predict(dval)
    xgb_score = roc_auc_score(y_test, ypred)

    # print "training random forest"
    # clf = RandomForestClassifier(n_estimators=1000,
                                     # n_jobs = -1,
                                     # class_weight = "auto",
                                     # max_depth = 10)
    # clf.fit(X_train, y_train)
    # rf_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    rf_score = 0
    return xgb_score, rf_score

def model(features, label, val_train, val_test, n_round):
    """
    - features: pandas DataFrame
    - label: pandas DataFrame from truth_train file
    - val_train: pandas DataFrame from enrollment_train file
    - val_test: pandas DataFrame from enrollment_test file
    - n_round: number of iteration for xgboost
    """
    xgb_score = []
    rf_score = []
    X = features.as_matrix()
    y = np.array(label[1])
    skf = StratifiedKFold(y, n_folds=5, shuffle=True)
    for train_idx, test_idx in skf:
        print "==================================="
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        xgb_auc, rf_auc = evaluate(X_train, y_train, X_test, y_test, n_round)
        xgb_score.append(xgb_auc)
        rf_score.append(rf_auc)

    print "==================================="
    X_train = features.ix[val_train['enrollment_id']]
    y = np.array(label[label[0].isin(val_train['enrollment_id'])][1])
    X_test = features.ix[val_test['enrollment_id']]
    y_test = np.array(label[label[0].isin(val_test['enrollment_id'])][1])
    xgb_auc, rf_auc = evaluate(X_train.as_matrix(), y,
                            X_test.as_matrix(), y_test, n_round)
    print "==================================="
    print '5_fold: %f (+/- %f), min: %f, max: %f' \
            % (np.mean(xgb_score), np.std(xgb_score), min(xgb_score), max(xgb_score))
    # print 'randomforest + 5_fold: %f, min: %f, max: %f' \
            # % (np.mean(rf_score), min(rf_score), max(rf_score))
    print 'xgboost + valid1: %f' % xgb_auc
    # print 'randomforest + valid1: %f' % rf_auc
    # print 'gbm + valid1: %f' % gbm_auc




from generate_feature import *
from model import *
import cPickle, csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

def loadans():
    trainy = []
    with open('/tmp2/kdd/truth_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            enrole_id = int(row[0])
            trainy.append(int(row[1]))
    return trainy

def outputans(ans, id_file_path, path):
    idxs = []
    with open(id_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'enrollment_id': continue
            idxs.append(int(row[0]))

    with open(path, 'w') as f:
        for i, enrole_id in enumerate(idxs):
            f.write(str(enrole_id) + ',' + str(ans[i]) + '\n')

with open('../../data/feature/train_feature.npy', 'r') as f:
    trainX = np.load(f)

with open('../../data/feature/blend_feature.npy', 'r') as f:
    blendX = np.load(f)

with open('../../data/feature/test_feature.npy', 'r') as f:
    testX = np.load(f)

print np.shape(trainX)
print np.shape(blendX)

#with open('fet.pkl', 'r') as f:
#    train_X, test_X = cPickle.load(f)

trainy = loadans()

#with open('./current/fet.pkl', 'r') as f:
#    train_X, test_X = cPickle.load(f)
#with open('fet.pkl', 'r') as f:
#    train_X, test_X = cPickle.load(f)

train_y = loadans()

clf = CalibratedClassifierCV(
    base_estimator=RandomForestClassifier(
            criterion='entropy',
            n_estimators = 3000,
            class_weight = "auto",
            max_depth = 12,
            n_jobs=-1
        )
    )

clf.fit(trainX, trainy)
outputans(clf.predict_proba(blendX)[:, 1],
    '/tmp2/kdd/enrollment_blend.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/preds/rf_66_blend.csv')
outputans(clf.predict_proba(testX)[:, 1],
    '/tmp2/b01902066/KDD/data/internal1/enrollment_test.csv',
    '/tmp2/b01902066/KDD/kdd15/blending/testpreds/rf_66_test.csv')

#clf.fit(train_X, train_y)
#
#print "done building trees"
#
#ret = []
#for est in clf.estimators_:
#    if ret == []:
#        ret = np.reshape(est.predict(train_X), (-1, 1))
#    else:
#        ret = np.hstack((ret, np.reshape(est.predict(train_X), (-1, 1))))
#
#valret = []
#for est in clf.estimators_:
#    if valret == []:
#        valret = np.reshape(est.predict(test_X), (-1, 1))
#    else:
#        valret = np.hstack((valret, np.reshape(est.predict(test_X), (-1, 1))))
#
##print clf.score(test_X, test_y)
##print roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1])
#
#dump_svmlight_file(ret, train_y, 'upRFtrain.svmlight', zero_based=False)
#dump_svmlight_file(valret, np.zeros(len(valret)), 'upRFtest.svmlight', zero_based=False)

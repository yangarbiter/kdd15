import numpy as np
import os, csv, subprocess, re, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.datasets import dump_svmlight_file
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.ensemble import GradientBoostingClassifier as  GBM
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed

PRED_PATH = './allpreds/625/'

class Blender():
    def __init__(self):
        self.clf = RF(n_estimators = 2000, n_jobs = -1, max_depth = 5, criterion = "entropy")
        #self.clf = Ada(base_estimator = DT(criterion = "entropy", max_depth = 1) ,n_estimators = 1000, learning_rate = 0.05)
        #self.clf = GBM(n_estimators = 1000, learning_rate = 0.05, max_depth = 2)
    def feature_select(self, X, y, valX, valy):
        A = []
        for i in range(X.shape[1]):
            self.clf.fit(np.delete(X, [i], 1),y)
            A.append(self.auc_score(np.delete(valX, [i], 1), valy))
        return max(A), A.index(max(A)), self.auc_score(np.delete(X, [A.index(max(A))], 1), y)

    def train(self, X, y, C=1.0, gamma=1.0):
        self.clf.fit(X, y)

    def ori_score(self, X, y):
        for i in range(len(X[0])):
            print roc_auc_score(y, X[:, i])

    def predict(self, X, y=[]):
        return self.clf.predict_proba(X)[:, 1]

    def score(self, X, y):
        return self.clf.score(X, y)

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict_proba(X)[:, 1])
    
    def print_weight(self):
        print self.clf.feature_importances_

def read_preds(path, filelist=[], verbose=True):
    preds = []
    valpreds = []
    testpreds = []
    if filelist == []:
        filelist = [filename for filename in sorted(os.listdir(path)) if '_blend.csv' in filename]

    for filename in filelist:
        if '_blend.csv' not in filename: continue

        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pred.append(float(row[1]))
        pred = calibrate(pred)
        preds.append(pred)
        if verbose:
            print filename
        assert np.shape(pred) == (20000,)

        val_filename = filename[:-9] + 'val.csv'
        pred = []
        with open(path+val_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pred.append(float(row[1]))
        pred = calibrate(pred)
        assert np.shape(pred) == (24108,)
        valpreds.append(pred)

        test_filename = filename[:-9] + 'test.csv'
        pred = []
        with open(path+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pred.append(float(row[1]))
        pred = calibrate(pred)
        assert np.shape(pred) == (80362,)
        testpreds.append(pred)

    return np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T

def calibrate(x):
    return (x-np.mean(x))/(np.max(x) - np.min(x))
    #x = np.array(x)
    #order = x.argsort()
    #ranks = order.argsort()
    #return ranks / np.float(len(ranks))

def read_truth():
    eid = []
    y = []
    with open('./truth_blend.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            eid.append(int(row[0]))
            y.append(float(row[1]))

    testeid = []
    testy = []
    with open('./truth_test.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            testeid.append(int(row[0]))
            testy.append(float(row[1]))

    return eid, np.array(y), testeid, np.array(testy)

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

def main():
    eid, y, valeid, valy = read_truth()
    X, valX, testX = read_preds(PRED_PATH)

    print np.shape(X), np.shape(valX), np.shape(testX)

    clf = Blender()
    clf.train(X, y)
    print 'auc in: ', clf.auc_score(X, y)
    print 'auc val:', clf.auc_score(valX, valy)
    #clf.print_weight()
    
    best = -np.inf
    score = -1
    while score > best:
        print X.shape, valX.shape, testX.shape
        best = score
        score, i, in_score = clf.feature_select(X, y, valX, valy)
        print 'delete i:', i
        print 'auc in:', in_score
        print 'auc val:', score
        if score > best:
            X = np.delete(X,[i], 1)
            valX = np.delete(valX, [i], 1)
            testX = np.delete(testX, [i], 1)
    
    
    clf.train(np.vstack((X, valX)), np.hstack((y, valy)) )

    outputans(clf.predict(testX), './enrollment_test.csv', './test_preds/RF.csv')

if __name__ == "__main__":
    main()

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
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from joblib import Parallel, delayed

PRED_PATH = './allpreds/626/'
def sigmoid(P):
    for i in range(len(P)):
        P[i] = 1.0/(1+np.exp(-P[i]))
    return P
class Uniform():
    def __init__(self):
        pass
    def predict(self, X, y = []):
        P = np.zeros((X.shape[0], X.shape[1]))
        P[:,1] = np.mean(X, axis = 1)
        return P
    def predict_proba(self, X, y = []):
        P = np.zeros((X.shape[0], X.shape[1]))
        P[:,1] = np.mean(X, axis = 1)
        P = sigmoid(P)
        return P

class Blender():
    def __init__(self):
        self.clf = Uniform()
    
    def feature_select(self, X, y):
        scores = []
        for i in range(X.shape[1]):
            scores.append(self.auc_score(np.delete(X, [i], 1), y))
        max_score = max(scores)
        max_index = scores.index(max_score)
        return max_score, max_index

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict(X)[:, 1])

def read_preds(path, filelist=[], verbose=True):
    preds = []
    valpreds = []
    testpreds = []
    F = []
    if filelist == []:
        filelist = [filename for filename in sorted(os.listdir(path)) if '_blend.csv' in filename]

    for filename in filelist:
        if '_blend.csv' not in filename: continue

        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if 'ID' in row:
                    continue
                pred.append(float(row[1]))
        pred = calibrate(pred)
        preds.append(pred)
        if verbose:
            print filename
        assert np.shape(pred) == (20000,)

        F.append(filename[:-10])
        val_filename = filename[:-9] + 'val.csv'
        pred = []
        with open(path+val_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if 'ID' in row:
                    continue
                pred.append(float(row[1]))
        pred = calibrate(pred)
        assert np.shape(pred) == (24108,)
        valpreds.append(pred)

        test_filename = filename[:-9] + 'test.csv'
        pred = []
        with open(path+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if 'ID' in row:
                    continue
                pred.append(float(row[1]))
        pred = calibrate(pred)
        assert np.shape(pred) == (80362,)
        testpreds.append(pred)

    return F, np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T

def calibrate(x):
    return (x-np.mean(x))/(np.max(x) - np.min(x))
    #x = np.array(x)
    #order = x.argsort()
    #ranks = order.argsort()
    #return ranks / np.float(len(ranks))
    #return x
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
    F,X, valX, testX = read_preds(PRED_PATH)

    X = np.vstack((X, valX))
    y = np.hstack((y, valy))
    print np.shape(X), np.shape(testX)

   # best_list = [3, 8, 37, 15, 9, 20, 21, 32, 17]
    best_list = [i for i in range(X.shape[1])]
    bestX = None
    for i in best_list:
        if bestX is None:
            bestX = X[:,i]
            bestX = bestX.reshape(-1, 1)
            best_testX = testX[:,i]
            best_testX = best_testX.reshape(-1, 1)
        else:
            bestX = np.hstack((bestX, X[:,i].reshape(-1, 1) )) 
            best_testX = np.hstack((best_testX, testX[:,i].reshape(-1, 1)))
    
    model = Blender()
    print 'AUC : ', model.auc_score(bestX, y)
    best = -np.inf
    score = -1
    while score > best:
        best = score
        print bestX.shape, best_testX.shape
        score, i= model.feature_select(bestX, y)
        print "DELETE MODEL : ",i, F[i]
        print 'AUC :', score
        if score > best:
            del F[i]
            bestX = np.delete(bestX,[i], 1)
            best_testX = np.delete(best_testX, [i], 1)
    
    print "Remaining models:"
    for f in F:
        print f
    """ 
    best = -np.inf
    score = -1
    while score > best:
        print bestX.shape,  best_testX.shape
        scores = [model.auc_score(np.hstack((bestX, X[:,i].reshape(-1, 1))) , y) for i in range(X.shape[1])] 
        score = max(scores)
        i = scores.index(score)
        #print "ADD MODEL : ",i, F[i]
        print '(ADDING) AUC :', score
        if score > best:
            bestX = np.hstack((bestX, X[:,i].reshape(-1, 1)))
            best_testX = np.hstack((best_testX, testX[:,i].reshape(-1, 1)))
    """
    outputans(model.clf.predict_proba(best_testX)[:,1], './enrollment_test.csv', './test_preds/626_uniform.csv')

if __name__ == "__main__":
    main()

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
import xgboost as xgb

PRED_PATH = './allpreds/625/'
"""
def xgb_train_eval(X, y, valX, valy): 
    dtrain = xgb.DMatrix( X , label = y)
    dval = xgb.DMatrix( valX, label = valy)
    param = {'booster': 'gbtree',
             'bst:max_depth':1,
             #'bst:eta': 0.001,
             'min_child_weight':100,
             'subsample':0.5,
             'lambda': 0.1,
             'gamma': 0,
             #'objective':'binary:logistic',
             'objective':'rank:pairwise',
             'eval_metric': 'auc',
             'silent': 1,
             'nthread': 1}
    plst = param.items()
    #evallist = [(dval, 'eval'), (dtrain, 'train')]
    evallist = []
    num_round = 50
    bst = xgb.train(plst, dtrain, num_round, evallist)
    ypred = bst.predict(dval)
    xgb_score = roc_auc_score(valy, ypred)
    return xgb_score
"""
class Blender():
    def __init__(self):
        #self.clf = RF(n_estimators = 200, n_jobs = -1, max_depth = 7, criterion = "entropy")
        #self.clf = Ada(base_estimator = DT(criterion = "entropy", max_depth = 1) ,n_estimators = 1000, learning_rate = 0.05)
        #self.clf = GBM(n_estimators = 1000, learning_rate = 0.05, max_depth = 2)
        pass
    """
    def feature_select(self, X, y, valX, valy):
        candidate_i = [i for i in range(X.shape[1])]
        A = Parallel(n_jobs=-1, backend="threading")(
                    delayed(xgb_train_eval)(np.delete(X, [i], 1), y, np.delete(valX, [i], 1), valy) \
                        for i in candidate_i)
        print A
        return max(A), A.index(max(A)), xgb_train_eval(np.delete(X, [A.index(max(A))], 1), y, np.delete(X, [A.index(max(A))],1 ), y)
    """
    def train(self, X, y, valX = None, valy = None):
        param = {'booster': 'gbtree',
                 'bst:max_depth':3,
                 'bst:eta': 0.01,
                 #'min_child_weight':100,
                 'subsample':0.8,
                 'lambda': 0.5,
                 'gamma': 0,
                 #'objective':'binary:logistic',
                 'objective':'rank:pairwise',
                 'eval_metric': 'auc',
                 'silent': 1,
                 'nthread': 1}
        
        num_round = 300
        dtrain = xgb.DMatrix(X, label = y)
        plst = param.items()
        if valX is None:
            evallist = []
        else:
            dval = xgb.DMatrix( valX, label = valy)
            evallist = [(dval, 'eval'), (dtrain, 'train')]
        bst = xgb.train(plst, dtrain, num_round, evallist)
        self.clf = bst 
    
    def ori_score(self, X, y):
        for i in range(len(X[0])):
            print roc_auc_score(y, X[:, i])

    def predict(self, X, y=[]):
        d = xgb.DMatrix(X)
        return self.clf.predict(d)
        #return self.clf.predict_proba(X)[:, 1]

    def auc_score(self, X, y):
        return roc_auc_score(y, self.predict(X))
    
    def print_weight(self):
        print self.clf.feature_importances_

def read_preds(path, filelist=[], verbose=True):
    F = []
    preds = []
    valpreds = []
    testpreds = []
    remove_list = [ "team3_110_GBM_md6_ne2800_lr0.01",
            "ada",
            "logis",
            "team1_[Vauc-0.901429-Avg2Cv0][Blend-BigF][119_106_13-0130][Nor-0][ADA][1500-0.01-4]",
            "team1_[Vauc-0.897567-Avg2Cv0][Blend-BigF][119_106_11-0133][Nor-2][LR][l2-1000.0]",
            "team1_[Vauc-0.902810-Avg2Cv0][Blend-BigF][119_106_s04-0133][Nor-0][ADA][1500-0.01-4]",
            "team2_xgb_107_rank",
            "team2_xgb"]
            

    if filelist == []:
        filelist = [filename for filename in sorted(os.listdir(path)) if '_blend.csv' in filename]
    
    for filename in filelist:
        if '_blend.csv' not in filename: continue
        """
        if filename[:-10] in remove_list:
            print "REMOVE :", filename[0:-10]
            continue
        """
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
        F.append(filename[:-10])
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
    F, X, valX, testX = read_preds(PRED_PATH)

    #print np.shape(X), np.shape(valX), np.shape(testX)

    clf = Blender()
    clf.train(X, y, valX, valy)
    print 'AUC Vn: ', clf.auc_score(X, y)
    print 'AUC Val:', clf.auc_score(valX, valy)
    #clf.print_weight()
    """
    best = -np.inf
    score = -1
    while score > best:
        print X.shape, valX.shape, testX.shape
        best = score
        score, i, in_score = clf.feature_select(X, y, valX, valy)
        print "DELETE MODEL : ",i, F[i]
        del F[i]
        print 'AUC in:', in_score
        print 'AUC val:', score
        if score > best:
            X = np.delete(X,[i], 1)
            valX = np.delete(valX, [i], 1)
            testX = np.delete(testX, [i], 1)
    """
    clf.train(np.vstack((X, valX)), np.hstack((y, valy)) )

    outputans(clf.predict(testX), './enrollment_test.csv', './test_preds/626_37models_xgboost.csv')

if __name__ == "__main__":
    main()

import cPickle
import numpy as np
import os, csv, subprocess, re, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed

SVM_RANK_PATH = './liblinear-ranksvm-1.95/'
PRED_PATH = './new_allpreds/best_4/'
TMP_PATH = '/tmp/kddtmp/'
def rank_SVM_train_eval(X, y, valX, valy, C):
    train_file = TMP_PATH+'blend' + str(C) + '.svmlight'
    model_file = TMP_PATH+'blending' + str(C) + '.model'
    preds_file = TMP_PATH+'blending_preds' + str(C) + '.pred'
    dump_svmlight_file(X, y, train_file, zero_based=False)
    result = subprocess.check_output(
            SVM_RANK_PATH + 'train ' +\
            '-c ' + str(C) + ' -s 8 ' +\
            train_file + ' ' + model_file,
            #'../current/svm_perf/svm_perf_learn -t 2 -g 0.01 -c 0.01 -l 10 -w 3 blend.svmlight blending.model',
            shell=True
        )
    val_file = TMP_PATH+'val' + str(C) + '.svmlight'
    dump_svmlight_file(valX, valy, val_file, zero_based=False)
    result = subprocess.check_output(
            SVM_RANK_PATH + 'predict ' +\
            val_file + ' ' + model_file + ' ' + preds_file,
            shell=True
        )
    with open(preds_file, 'r') as f:
        preds = [float(line) for line in f.readlines()]

    return roc_auc_score(valy, preds)
    #return float(re.findall('ROCArea[\s:0-9\.]+', result)[0][11:])/100.0

def rank_SVM_feature_select_eval(X, y, valX, valy, C, i):
    train_file = TMP_PATH+'blend.svmlight'
    val_file = TMP_PATH+'val.svmlight'
    model_file = TMP_PATH+'blending' + str(i) + '.model'
    preds_file = TMP_PATH+'blending_preds' + str(i) + '.pred'
    result = subprocess.check_output(
            SVM_RANK_PATH + 'train ' +\
            '-c ' + str(C) + ' -s 8 ' + '-r ' +str(i+1)+' '+\
            train_file + ' ' + model_file,
            #'../current/svm_perf/svm_perf_learn -t 2 -g 0.01 -c 0.01 -l 10 -w 3 blend.svmlight blending.model',
            shell=True
        )
    result = subprocess.check_output(
            SVM_RANK_PATH + 'predict ' +\
            '-r ' +str(i+1)+' '+\
            val_file + ' ' + model_file + ' ' + preds_file,
            shell=True
        )
    with open(preds_file, 'r') as f:
        preds = [float(line) for line in f.readlines()]

    return roc_auc_score(valy, preds)
    #return float(re.findall('ROCArea[\s:0-9\.]+', result)[0][11:])/100.0

class Blender():
    def __init__(self):
        pass

    def grid_search(self, X, y, valX, valy):
        candidate_c = [10**i for i in range(-10, 10)]
        res = Parallel(n_jobs=-1, backend="threading")(
                    delayed(rank_SVM_train_eval)(X, y, valX, valy, c) \
                        for c in candidate_c)
        print "grid search result: ", res
        return max(res), candidate_c[res.index(max(res))]
    def feature_select(self, X, y, valX, valy, C):
        train_file = TMP_PATH+'blend.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        val_file = TMP_PATH+'val.svmlight'
        dump_svmlight_file(valX, valy, val_file, zero_based=False)
        candidate_i = [i for i in range(X.shape[1])]
        
        #for i in candidate_i:
        #    rank_SVM_feature_select_eval(X, y, valX, valy, C, i)
        res = Parallel(n_jobs=-1, backend="threading")(
                    delayed(rank_SVM_feature_select_eval)(X, y, valX, valy, C, i) \
                        for i in candidate_i)
        #print "feature select result: ", res
        return max(res), candidate_i[res.index(max(res))]

    def train(self, X, y, C=1.0, gamma=1.0):
        dump_svmlight_file(X, y, 'blend.svmlight', zero_based=False)
        result = subprocess.check_output(
                SVM_RANK_PATH + 'train -c ' + str(C) + ' -s 8 blend.svmlight blending.model',
                shell=True
            )

    def ori_score(self, X, y):
        P = []
        for i in range(len(X[0])):
            P.append(roc_auc_score(y, X[:, i]))
        return P


    def predict(self, X, y=[]):
        if y != []:
            dump_svmlight_file(X, y, 'val.svmlight', zero_based=False)
            result = subprocess.check_output(
                    SVM_RANK_PATH + 'predict val.svmlight ./blending.model ./val_prediction',
                    shell=True
                )
            return re.findall('ROCArea[\s:0-9\.]+', result)[0][11:].strip()
        else:
            return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict_proba(X)[:, 1])

def read_preds(path, filelist=[], verbose=True):
    F = []
    preds = []
    valpreds = []
    testpreds = []
    if filelist == []:
        filelist = [filename for filename in sorted(os.listdir(path)) if '_blend.csv' in filename]
    if len(sys.argv) >= 2:
        print 'deleted ', filelist[int(sys.argv[1])]
        del filelist[int(sys.argv[1])]

    for filename in filelist:
        if '_blend.csv' not in filename: continue

        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if "ID" in row:
                    continue
                pred.append(float(row[1]))
        #print(len(pred))
        #pred = calibrate(pred)
        preds.append(pred)
        if verbose:
            print filename

        F.append(filename[:-10])
        val_filename = filename[:-9] + 'val.csv'
        pred = []
        with open(path+val_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if "ID" in row:
                    continue
                pred.append(float(row[1]))
        #pred = calibrate(pred)
        valpreds.append(pred)

        test_filename = filename[:-9] + 'test.csv'
        pred = []
        with open(path+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if "ID" in row:
                    continue
                pred.append(float(row[1]))
        #pred = calibrate(pred)
        assert np.shape(pred) == (80362,)
        testpreds.append(pred)

    return F, np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T

def calibrate(X):
    new_X = []
    #return (x-np.mean(x))/(np.max(x) - np.min(x))
    for i in range(X.shape[1]):
        x = np.array(X[:,i])
        order = x.argsort()
        ranks = order.argsort()
        new_X.append(ranks / np.float(len(ranks)))
    return np.array(new_X).T
def read_truth():
    #with open('/tmp2/b01902066/KDD/kdd15/current/data/0610/label_train+blend+valid.npy', 'rb') as f:
    #    y = np.load(f)
    #return y
    eid = []
    y = []
    with open('./new_truth_blend.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            eid.append(int(row[0]))
            y.append(float(row[1]))

    testeid = []
    testy = []
    with open('./new_truth_test.csv', 'rb') as csvfile:
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

    if ans != []:
        with open(path, 'w') as f:
            for i, enrole_id in enumerate(idxs):
                f.write(str(enrole_id) + ',' + str(ans[i]) + '\n')
    else:
        with open('./blending_prediction', 'r') as f:
            preds = [float(i.strip()) for i in f.readlines()]
            preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))

            with open('./test_preds/710_ranksvm.csv', 'w') as fo:
                for i, ans in zip(idxs, preds):
                    fo.write(str(i) + ',' + str(ans) + '\n')

def main():
    eid, y, valeid, valy = read_truth()
    F, raw_X, raw_valX, raw_testX = read_preds(PRED_PATH)

    X = calibrate(raw_X)
    valX = calibrate(raw_valX)
    testX = calibrate(raw_testX)
    ori_list = [i for i in range(X.shape[1])]
    best_list = ori_list[:]

    clf = Blender()
    C = 0.0001
    print X.shape, valX.shape, testX.shape
    print 'auc val: ', rank_SVM_train_eval(X, y, valX, valy, C)
    
    """
    score, C = clf.grid_search(X, y, valX, valy)
    print 'C:', C
    print 'auc val:', score
    """
    best = -np.inf
    score = - 1
    while X.shape[1] > 1:
        score, i = clf.feature_select(X, y, valX, valy, C)
        print 'DELETE MODEL :', F[i]
        del F[i]
        del ori_list[i]
        print 'AUC val:', score
        X = np.delete(X,[i], 1)
        valX = np.delete(valX, [i], 1)
        testX = np.delete(testX, [i], 1)
        if score > best:
            best = score
            best_list = ori_list[:]
        print '=============================="'
        print "Remaining : ", X.shape, valX.shape, testX.shape
    
    cPickle.dump(best_list, open("./bestX.pkl", "w"))

    best_list = cPickle.load(open("./bestX.pkl", "r"))
    X = raw_X
    valX = raw_valX
    testX = raw_testX
    best_X = None
    best_valX = None
    best_testX = None
    for i in best_list:
        if best_X is None:
            best_X = X[:,i].reshape(-1, 1)
            best_valX = valX[:,i].reshape(-1, 1)
            best_testX = testX[:,i].reshape(-1, 1)
        else:
            best_X = np.hstack((best_X, X[:,i].reshape(-1, 1) )) 
            best_valX = np.hstack((best_valX, valX[:,i].reshape(-1, 1) )) 
            best_testX = np.hstack((best_testX, testX[:,i].reshape(-1, 1)))

    print best_X.shape, best_valX.shape, best_testX.shape

    best_X = calibrate(best_X)
    best_valX = calibrate(best_valX)
    best_testX = calibrate(best_testX)
    print 'auc val: ', rank_SVM_train_eval(best_X, y, best_valX, valy, C) 
    

    clf.train(np.vstack((best_X, best_valX)), np.hstack((y, valy)), C)
    #clf.train(best_valX, valy, C)
    #clf.train(bestX, y, C)


    if len(sys.argv) == 1:
        dump_svmlight_file(best_testX, np.zeros((len(best_testX))), 'test.svmlight', zero_based=False)
        result = subprocess.check_output(
            SVM_RANK_PATH+'predict test.svmlight ./blending.model ./blending_prediction',
            shell=True
        )
        outputans([], './enrollment_test.csv', '')

if __name__ == "__main__":
    main()


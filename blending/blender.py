
import numpy as np
import os, csv, subprocess, re, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed

SVM_PERF_PATH = '/tmp2/b01902066/KDD/kdd15/current/svm_perf/'
PRED_PATH = '/tmp2/b01902066/KDD/kdd15/blending/allpreds/625/'

def perf_SVM_CV(X, y, C, cv):
    ret = []
    train_file = 'blend' + str(C) + '.svmlight'
    model_file = 'blending' + str(C) + '.model'
    preds_file = 'blending_preds' + str(C) + '.pred'

    skf = StratifiedKFold(y, n_folds=cv, shuffle=True)
    for train_idx, test_idx in skf:
        trainX = X[train_idx]
        trainy = y[train_idx]
        testX = X[test_idx]
        testy = y[test_idx]

        dump_svmlight_file(trainX, trainy, train_file, zero_based=False)
        result = subprocess.check_output(
                SVM_PERF_PATH + 'svm_perf_learn ' +\
                '-c ' + str(C) + ' -l 10 -w 3 ' +\
                train_file + ' ' + model_file,
                #'../current/svm_perf/svm_perf_learn -t 2 -g 0.01 -c 0.01 -l 10 -w 3 blend.svmlight blending.model',
                shell=True
            )

        val_file = 'val' + str(C) + '.svmlight'
        dump_svmlight_file(testX, testy, val_file, zero_based=False)
        result = subprocess.check_output(
                SVM_PERF_PATH + 'svm_perf_classify ' +\
                val_file + ' ' + model_file + ' ' + preds_file,
                shell=True
            )

        preds = []
        with open(preds_file, 'r') as f:
            preds = [float(line) for line in f.readlines()]

        ret.append(roc_auc_score(testy, preds))

    return np.mean(ret)

def perf_SVM_train_eval(X, y, valX, valy, C):
    train_file = 'blend' + str(C) + '.svmlight'
    model_file = 'blending' + str(C) + '.model'
    preds_file = 'blending_preds' + str(C) + '.pred'
    dump_svmlight_file(X, y, train_file, zero_based=False)
    result = subprocess.check_output(
            SVM_PERF_PATH + 'svm_perf_learn ' +\
            '-c ' + str(C) + ' -l 10 -w 3 ' +\
            train_file + ' ' + model_file,
            #'../current/svm_perf/svm_perf_learn -t 2 -g 0.01 -c 0.01 -l 10 -w 3 blend.svmlight blending.model',
            shell=True
        )

    val_file = 'val' + str(C) + '.svmlight'
    dump_svmlight_file(valX, valy, val_file, zero_based=False)
    result = subprocess.check_output(
            SVM_PERF_PATH + 'svm_perf_classify ' +\
            val_file + ' ' + model_file + ' ' + preds_file,
            shell=True
        )

    preds = []
    with open(preds_file, 'r') as f:
        preds = [float(line) for line in f.readlines()]

    return roc_auc_score(valy, preds)
    #return float(re.findall('ROCArea[\s:0-9\.]+', result)[0][11:])/100.0


class Blender():
    def __init__(self):
        pass

    def grid_search(self, X, y, valX, valy):
        candidate_c = [10**i for i in range(0, 4)]
        res = Parallel(n_jobs=len(candidate_c), backend="threading")(
                    delayed(perf_SVM_train_eval)(X, y, valX, valy, c) \
                        for c in candidate_c)
        #res = Parallel(n_jobs=len(candidate_c), backend="threading")(
        #            delayed(perf_SVM_CV)(X, y, c, 10) \
        #                for c in candidate_c)
        print "grid search result: ", res
        return max(res), candidate_c[res.index(max(res))]

    def train(self, X, y, C=1.0, gamma=1.0):
        dump_svmlight_file(X, y, 'blend.svmlight', zero_based=False)
        result = subprocess.check_output(
                SVM_PERF_PATH + 'svm_perf_learn -c ' + str(C) + ' -l 10 -w 3 blend.svmlight blending.model',
                shell=True
            )

    def ori_score(self, X, y):
        for i in range(len(X[0])):
            print roc_auc_score(y, X[:, i])


    def predict(self, X, y=[]):
        if y != []:
            dump_svmlight_file(X, y, 'val.svmlight', zero_based=False)
            result = subprocess.check_output(
                    SVM_PERF_PATH + 'svm_perf_classify val.svmlight ./blending.model ./val_prediction',
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
    preds = []
    valpreds = []
    testpreds = []
    if filelist == []:
        filelist = [filename for filename in sorted(os.listdir(path)) if '_blend.csv' in filename]
    if len(sys.argv) >= 2:
        for idx in sys.argv[1:]:
            print 'exclude ', filelist[int(idx)]
            del filelist[int(idx)]

    for filename in filelist:
        if '_blend.csv' not in filename: continue

        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pred.append(float(row[1]))
        #print(len(pred))
        pred = calibrate(pred)
        preds.append(pred)
        if verbose:
            print filename

        try:
            assert np.shape(pred) == (20000,)
        except:
            print np.shape(pred)

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

    return np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T, filelist

def calibrate(x):
    x = np.array(x)
    order = x.argsort()
    ranks = order.argsort()
    return ranks / np.float(len(ranks))
    #return (x - np.min(x)) / (np.max(x) - np.min(x))

def read_truth():
    #with open('/tmp2/b01902066/KDD/kdd15/current/data/0610/label_train+blend+valid.npy', 'rb') as f:
    #    y = np.load(f)
    #return y
    eid = []
    y = []
    with open('/tmp2/kdd/truth_blend.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            eid.append(int(row[0]))
            y.append(float(row[1]))

    testeid = []
    testy = []
    with open('/tmp2/b01902066/KDD/data/internal1/truth_test.csv', 'rb') as csvfile:
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

            with open('./perfsvm_prediction', 'w') as fo:
                for i, ans in zip(idxs, preds):
                    fo.write(str(i) + ',' + str(ans) + '\n')

def parse_exp_result(res):
    acc = []
    #print res.split('\n')
    for line in res.split('\n')[:-1]:
        acc.append(float(line.split(':')[1]))
    return acc

def main():
    filelist = []
    removed = []
    #with open('/tmp2/b01902066/KDD/kdd15/blending/removed', 'r') as f:
    #    removed = [filename.strip() for filename in f.readlines()]
    for filename in sorted(os.listdir(PRED_PATH)):
        if '_blend.csv' in filename and\
            filename not in removed and\
            'ada_blend' not in filename:
            filelist.append(filename)

    eid, y, valeid, valy = read_truth()
    X, valX, testX, filelist = read_preds(PRED_PATH, filelist)

    print np.shape(X), np.shape(valX), np.shape(testX)

    clf = Blender()
    score, C = clf.grid_search(X, y, valX, valy)
    print 'C:', C
    print 'auc val:', score
    print perf_SVM_train_eval(X, y, valX, valy, C)
    clf.ori_score(X, y)
    print "------------------"
    clf.ori_score(valX, valy)
    print "------------------"
    #clf.train(X, y, C)
    clf.train(np.vstack((X, valX)), np.hstack((y, valy)), C)
    #print 'auc in: ', clf.predict(X, y)
    #print 'auc val:', clf.predict(valX, valy)

    #dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
    #result = subprocess.check_output(
    #    SVM_PERF_PATH+'svm_perf_classify test.svmlight ./blending.model ./blending_prediction',
    #    shell=True
    #)
    #outputans([], '/tmp2/b01902066/KDD/data/enrollment_test.csv', '')
    #exit()

    if len(sys.argv) == 1:

        os.chdir('./exp')
        for i in range(1):
            result = subprocess.check_output(
                    './run_exps.sh ' + str(len(X[0])-i),
                    shell=True
                )
            result = subprocess.check_output(
                    './view_result.sh ' + str(len(X[0])-i),
                    shell=True
                )
            res = parse_exp_result(result)
            #print res

            with open('../blending.model', 'r') as f:
                w = f.readlines()[-1].split()[1:-1]
            ind = np.argsort(res)[::-1]
            for i in ind:
                print (filelist[i], res[i], w[i].split(':')[1])
            #print 'remove:', filelist[ind[0]]
            #print 'max auc:', res[ind[0]]
            #os.remove(PRED_PATH + filelist[ind[0]])
            del filelist[ind[0]]
        os.chdir('../')

        #dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
        #result = subprocess.check_output(
        #    SVM_PERF_PATH+'svm_perf_classify test.svmlight ./blending.model ./blending_prediction',
        #    shell=True
        #)
        #outputans([], '/tmp2/b01902066/KDD/data/enrollment_test.csv', '')

    #outputans(clf.predict(testX)[:, 1],
    #        '/tmp2/b01902066/KDD/data/enrollment_test.csv',
    #        '610_gbm_gbmrank_rf_ada_polylog_blend_logist_ans.csv')

if __name__ == "__main__":
    main()


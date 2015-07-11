
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

SVM_PERF_PATH = '/tmp2/b01902066/KDD/kdd15/current/svm_perf'
PRED_PATH = '/tmp2/b01902066/KDD/kdd15/blending/allpreds/708'
#PRED_PATH = '/tmp2/kddblending/new_blending1/with_danger/'
SVM_RANK_PATH = '/tmp2/b01902066/KDD/kdd15/blending/liblinear-ranksvm-1.95'
DATA_PATH = '/tmp2/b01902066/KDD/kdd15/blending/data'
TMPFS_PATH = '/tmp/b01902066'

def perf_SVM_train_eval(train_file, val_file, valy, C, r=0):
    model_file = 'blending_' + str(C) + '_' + str(r) + '.model'
    preds_file = 'blending_preds' + str(C) + '_' + str(r) +'.pred'
    RANKSVM_TRAIN_CMD = SVM_RANK_PATH + '/train ' + '-c ' + str(C) + ' -s 8 '
    RANKSVM_PRED_CMD = SVM_RANK_PATH + '/predict '
    if r > 0:
        RANKSVM_TRAIN_CMD += '-r ' + str(r) + ' '
        RANKSVM_PRED_CMD += '-r ' + str(r) + ' '

    result = subprocess.check_output(
            #SVM_PERF_PATH + '/svm_perf_learn ' +\
            #'-c ' + str(C) + ' -l 10 -w 3 ' +\
            RANKSVM_TRAIN_CMD + train_file + ' ' + model_file,
            #'../current/svm_perf/svm_perf_learn -t 2 -g 0.01 -c 0.01 -l 10 -w 3 blend.svmlight blending.model',
            shell=True
        )

    result = subprocess.check_output(
            #SVM_PERF_PATH + '/svm_perf_classify ' +\
            RANKSVM_PRED_CMD +
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
        train_file = './blend.svmlight'
        val_file = './val.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        dump_svmlight_file(valX, valy, val_file, zero_based=False)
        candidate_c = [10**i for i in range(-5, 5)]
        res = Parallel(n_jobs=len(candidate_c), backend="threading")(
                    delayed(perf_SVM_train_eval)(
                        train_file, val_file, valy, C=c)\
                        for c in candidate_c)
        #res = Parallel(n_jobs=len(candidate_c), backend="threading")(
        #            delayed(perf_SVM_CV)(X, y, c, 10) \
        #                for c in candidate_c)
        #print "grid search result: ", res
        return max(res), candidate_c[res.index(max(res))]

    def removal_select(self, X, y, valX, valy, C):
        train_file = './blend.svmlight'
        val_file = './val.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        dump_svmlight_file(valX, valy, val_file, zero_based=False)

        candidate_i = [i for i in range(X.shape[1])]
        res = Parallel(n_jobs=-1, backend="threading")(
                delayed(perf_SVM_train_eval)\
                    (train_file, val_file, valy, C=C, r=i+1)\
                    for i in candidate_i
            )
        return max(res), candidate_i[res.index(max(res))]

    def addition_select(self, oriX, X, y, orivalX, valX, valy, C):
        #C = C*len(X)/100.
        train_file = './blend.svmlight'
        val_file = './val.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        dump_svmlight_file(valX, valy, val_file, zero_based=False)

        candidate_i = [i for i in range(chidxs)]
        res = Parallel(n_jobs=-1, backend="threading")(
                delayed(perf_SVM_train_eval)\
                    (train_file, val_file, valy, C=C, r=i+1)\
                    for i in candidate_i
            )
        return max(res), candidate_i[res.index(max(res))]

    def train(self, X, y, C=1.0, gamma=1.0):
        train_file = './blend.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        result = subprocess.check_output(
                SVM_RANK_PATH + '/train ' +\
                '-c ' + str(C) + ' -s 8 ' +\
                train_file + ' blending.model',
                shell=True
            )

    def ori_score(self, X, y, valX, valy):
        ret = []
        for i in range(len(X[0])):
            print roc_auc_score(y, X[:, i]), roc_auc_score(valy, valX[:, i])
            ret.append([roc_auc_score(y, X[:, i]), roc_auc_score(valy, valX[:, i])])
        return ret

    def predict(self, X, y=[]):
        if y != []:
            val_file = TMPFS_PATH + '/val.svmlight'
            dump_svmlight_file(X, y, val_file, zero_based=False)
            result = subprocess.check_output(
                    #SVM_PERF_PATH + '/svm_perf_classify' +\
                    SVM_RANK_PATH + '/predict ' +\
                    val_file + ' ./blending.model ./val_prediction',
                    shell=True
                )
            preds = []
            with open('val_prediction', 'r') as f:
                preds = [float(line) for line in f.readlines()]
            return preds
        else:
            return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict_proba(X)[:, 1])

def read_preds(path, filelist=[], verbose=False):
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
        with open(path+'/'+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if row[0] == 'ID': continue
                pred.append(float(row[1]))
        #print(len(pred))
        try:
            assert np.shape(pred) == (20000,)
            #assert np.shape(pred) == (19286,)
        except:
            print np.shape(pred)

        preds.append(rank_calibrate(pred))
        if verbose:
            print filename


        val_filename = filename[:-9] + 'val.csv'
        pred = []
        with open(path+'/'+val_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if row[0] == 'ID': continue
                pred.append(float(row[1]))
        assert np.shape(pred) == (24108,)
        valpreds.append(rank_calibrate(pred))

        test_filename = filename[:-9] + 'test.csv'
        pred = []
        with open(path+'/'+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if row[0] == 'ID': continue
                pred.append(float(row[1]))
        assert np.shape(pred) == (80362,)
        testpreds.append(rank_calibrate(pred))

    return np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T, filelist

def rank_calibrate(x):
    x = np.array(x)
    order = x.argsort()
    ranks = order.argsort()
    return ranks / np.float(len(ranks))

def read_truth():
    eid = []
    y = []
    with open(DATA_PATH + '/truth_blend.csv', 'rb') as csvfile:
    #with open('/tmp2/kddblending/new_blend_data1/blend1/truth_test.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            eid.append(int(row[0]))
            y.append(float(row[1]))

    testeid = []
    testy = []
    with open(DATA_PATH + '/internal1/truth_test.csv', 'rb') as csvfile:
    #with open('/tmp2/kddblending/new_blend_data1/valid1/truth_test.csv', 'rb') as csvfile:
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
    for line in res.split('\n')[:-1]:
        acc.append(float(line.split(':')[1]))
    return acc

def main():
    filelist = []
    removed = []

    #with open('/tmp2/b01902066/KDD/kdd15/blending/selected', 'r') as f:
    #    filelist = [filename.strip() for filename in f.readlines()]

    with open('/tmp2/b01902066/KDD/kdd15/blending/removed', 'r') as f:
        removed = [filename.strip() for filename in f.readlines()]

    for filename in sorted(os.listdir(PRED_PATH)):
        if '_blend.csv' in filename and filename not in removed:
            filelist.append(filename)

    sample_idx = np.random.randint(0, 2, (len(filelist)))
    temp = []
    for i, r in enumerate(sample_idx):
        if r == 1:
            temp.append(filelist[i])
    filelist = temp

    eid, y, valeid, valy = read_truth()
    X, valX, testX, filelist = read_preds(PRED_PATH, filelist)

    clf = Blender()
    #print "------------------"
    #ret = np.array(clf.ori_score(X, y, valX, valy))

    #dif = np.abs(ret[:, 0] - ret[:, 1])
    #dif = dif < 0.0015

    #valacc = ret[:, 1] > 0

    #filt = np.logical_and(dif, valacc)

    #print np.sum(filt)
    #for i, f in enumerate(filt):
    #    if not f:
    #        print filelist[i]
    #print "------------------"
    #X = X[:, filt]
    #valX = valX[:, filt]
    #testX = testX[:, filt]

    score, C = clf.grid_search(X, y, valX, valy)
    #C = 0.00001
    #print 'C:', C
    #print 'auc val:', score
    #print "------------------"
    #clf.train(valX, valy, C)
    #print roc_auc_score(y, clf.predict(X, y))

    #clf.train(np.vstack((X, valX)), np.hstack((y, valy)), C)

    #dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
    #result = subprocess.check_output(
    #    SVM_RANK_PATH + '/predict test.svmlight ./blending.model ./blending_prediction',
    #    shell=True
    #)
    ##outputans([], DATA_PATH + '/enrollment_test.csv', '')
    #outputans([], '/tmp2/b01902066/KDD/data/enrollment_test.csv', '')
    #exit()

    best = -np.inf
    score = -1
    while score > best:
        best = score
        score, i = clf.removal_select(X, y, valX, valy, C)
        print 'delete i:', filelist[i], 'auc val:', score

        X = np.delete(X, [i], 1)
        valX = np.delete(valX, [i], 1)
        testX = np.delete(testX, [i], 1)
        del filelist[i]
    print ''
    exit()

    clf.train(np.vstack((X, valX)), np.hstack((y, valy)), C)
    dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
    result = subprocess.check_output(
        SVM_RANK_PATH + '/predict test.svmlight ./blending.model ./blending_prediction',
        shell=True
    )
    outputans([], DATA_PATH + '/enrollment_test.csv', '')
    exit()

if __name__ == "__main__":
    main()


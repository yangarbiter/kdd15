
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
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed

SVM_PERF_PATH = '/tmp2/b01902066/KDD/kdd15/current/svm_perf'
#PRED_PATH = '/tmp2/b01902066/KDD/kdd15/blending/allpreds/710'
PRED_PATH = '/tmp2/kddblending/new_blending1/with_danger/'
SVM_RANK_PATH = '/tmp2/b01902066/KDD/kdd15/blending/liblinear-ranksvm-1.95'
DATA_PATH = '/tmp2/b01902066/KDD/kdd15/blending/data'
TMPFS_PATH = '/tmp/b01902066'

def perf_SVM_CV(X, y, C, cv, r=0):
    ret = []
    train_file = './blend' + str(C) + str(r) + '.svmlight'
    val_file = './val' + str(C) + str(r) + '.svmlight'
    model_file = 'blending' + str(C) + str(r) + '.model'
    preds_file = 'blending_preds' + str(C) + str(r) + '.pred'

    RANKSVM_TRAIN_CMD = SVM_RANK_PATH + '/train ' + '-c ' + str(C) + ' -s 8 '
    RANKSVM_PRED_CMD = SVM_RANK_PATH + '/predict '
    if r > 0:
        RANKSVM_TRAIN_CMD += '-r ' + str(r) + ' '
        RANKSVM_PRED_CMD += '-r ' + str(r) + ' '

    skf = StratifiedKFold(y, n_folds=cv, shuffle=False)
    for train_idx, test_idx in skf:
        trainX = X[train_idx]
        trainy = y[train_idx]
        testX = X[test_idx]
        testy = y[test_idx]

        dump_svmlight_file(trainX, trainy, train_file, zero_based=False)
        result = subprocess.check_output(
            RANKSVM_TRAIN_CMD + train_file + ' ' + model_file,
            shell=True
        )

        dump_svmlight_file(testX, testy, val_file, zero_based=False)
        result = subprocess.check_output(
            RANKSVM_PRED_CMD +
            val_file + ' ' + model_file + ' ' + preds_file,
            shell=True
        )

        preds = []
        with open(preds_file, 'r') as f:
            preds = [float(line) for line in f.readlines()]

        ret.append(roc_auc_score(testy, preds))

    return np.mean(ret)

def perf_SVM_train_eval(train_file, val_file, valy, C, r=0):
    #C = C*len(X)/100.
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
                        train_file, val_file, valy, c)\
                        for c in candidate_c)
        #res = Parallel(n_jobs=len(candidate_c), backend="threading")(
        #            delayed(perf_SVM_CV)(X, y, c, 10) \
        #                for c in candidate_c)
        print "grid search result: ", res
        return max(res), candidate_c[res.index(max(res))]

    def removal_select(self, X, y, valX, valy, C):
        #C = C*len(X)/100.
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
        #res = Parallel(n_jobs=-1, backend="threading")(
        #        delayed(perf_SVM_CV)\
        #            (X, y, C, 5, r=i+1)\
        #            for i in candidate_i
        #    )
        #print res
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
        #C = C*len(X)/100.
        train_file = './blend.svmlight'
        dump_svmlight_file(X, y, train_file, zero_based=False)
        result = subprocess.check_output(
                #SVM_PERF_PATH + '/svm_perf_learn' +\
                #'-c' + str(C) + ' -l 10 -w 3 blend.svmlight blending.model',
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
        with open(path+'/'+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if row[0] == 'ID': continue
                pred.append(float(row[1]))
        #print(len(pred))
        try:
            #assert np.shape(pred) == (20000,)
            assert np.shape(pred) == (19286,)
        except:
            print np.shape(pred)
        #preds.append(pred)
        #preds.append(rank_calibrate(pred))
        preds.append(minmax_calibrate(pred))
        #preds.append(std_calibrate(pred))
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
        #valpreds.append(pred)
        #valpreds.append(rank_calibrate(pred))
        valpreds.append(minmax_calibrate(pred))
        #valpreds.append(std_calibrate(pred))

        test_filename = filename[:-9] + 'test.csv'
        pred = []
        with open(path+'/'+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if row[0] == 'ID': continue
                pred.append(float(row[1]))
        assert np.shape(pred) == (80362,)
        #testpreds.append(pred)
        #testpreds.append(rank_calibrate(pred))
        testpreds.append(minmax_calibrate(pred))
        #testpreds.append(std_calibrate(pred))

    return np.array(preds).T, np.array(valpreds).T, np.array(testpreds).T, filelist

def rank_calibrate(x):
    x = np.array(x)
    order = x.argsort()
    ranks = order.argsort()
    return ranks / np.float(len(ranks))

def minmax_calibrate(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
    #t = np.log(((x - np.min(x)) / (np.max(x) - np.min(x)))+1)
    #return ((t - np.min(t)) / (np.max(t) - np.min(t)))

def std_calibrate(x):
    return (x - np.mean(x)) / np.var(x)

def isotonic_calibrate(x):
    return (x - np.mean(x)) / np.var(x)

def read_truth():
    #with open('/tmp2/b01902066/KDD/kdd15/current/data/0610/label_train+blend+valid.npy', 'rb') as f:
    #    y = np.load(f)
    #return y
    eid = []
    y = []
    #with open(DATA_PATH + '/truth_blend.csv', 'rb') as csvfile:
    with open('/tmp2/kddblending/new_blend_data1/blend1/truth_test.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            eid.append(int(row[0]))
            y.append(float(row[1]))

    testeid = []
    testy = []
    #with open(DATA_PATH + '/internal1/truth_test.csv', 'rb') as csvfile:
    with open('/tmp2/kddblending/new_blend_data1/valid1/truth_test.csv', 'rb') as csvfile:
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

    #with open('/tmp2/b01902066/KDD/kdd15/blending/selected', 'r') as f:
    #    filelist = [filename.strip() for filename in f.readlines()]

    with open('/tmp2/b01902066/KDD/kdd15/blending/removed', 'r') as f:
        removed = [filename.strip() for filename in f.readlines()]

    for filename in sorted(os.listdir(PRED_PATH)):
        if '_blend.csv' in filename and filename not in removed and 'team3' not in filename:
            filelist.append(filename)

    eid, y, valeid, valy = read_truth()
    X, valX, testX, filelist = read_preds(PRED_PATH, filelist)

    #trainX = []
    #trainy = []
    #for filename in filelist:
    #    x = []
    #    with open('/tmp2/kddtrainpred/' + filename[6:-4] + '_train.csv', 'rb') as csvfile:
    #        reader = csv.reader(csvfile, delimiter=",")
    #        for row in reader:
    #            x.append(float(row[1]))
    #    trainX.append(minmax_calibrate(x))

    #with open('/tmp2/kdd/truth_train.csv', 'rb') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=",")
    #    for row in reader:
    #        trainy.append(float(row[1]))

    #trainX = np.array(trainX).T
    #trainy = np.array(trainy).T

    #print trainX.shape
    #print trainy.shape

    #for i in range(X.shape[1]):
    #    iso = IsotonicRegression()
    #    iso.fit(trainX[:, i], trainy)
    #    X[:, i] = iso.transform(X[:, i])
    #    valX[:, i] = iso.transform(valX[:, i])


    #print X[0].tolist()

    #X = np.hstack((X, X**2))
    #valX = np.hstack((valX, valX**2))
    #testX = np.hstack((testX, testX**2))

    print np.shape(X), np.shape(valX), np.shape(testX)

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
    print 'C:', C
    print 'auc val:', score
    print "------------------"
    clf.train(valX, valy, C)
    print roc_auc_score(y, clf.predict(X, y))

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
    #while score > best:
    for k in range(0, np.shape(X)[1]):
        best = score
        score, i = clf.removal_select(X, y, valX, valy, C)
        print 'delete i:', filelist[i], 'auc val:', score

        X = np.delete(X, [i], 1)
        valX = np.delete(valX, [i], 1)
        testX = np.delete(testX, [i], 1)
        del filelist[i]

    #idxs = []
    #for k in range(np.shape(X)[1], -1, -1):
    #    score, i = clf.addition_select(X, y, valX, valy, C, k)
    #    print 'added i:', filelist[i], 'auc val:', score

    #    X = np.append(X, X[:,i])
    #    valX = np.append(valX, valX[:,i])
    #    testX = np.append(testX, testX[:,i])

    #    X = np.delete(X, [i], 1)
    #    valX = np.delete(valX, [i], 1)
    #    testX = np.delete(testX, [i], 1)
    #    del filelist[i]


    clf.train(np.vstack((X, valX)), np.hstack((y, valy)), C)
    dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
    result = subprocess.check_output(
        #SVM_PERF_PATH + '/svm_perf_classify test.svmlight ./blending.model ./blending_prediction',
        SVM_RANK_PATH + '/predict test.svmlight ./blending.model ./blending_prediction',
        shell=True
    )
    outputans([], DATA_PATH + '/enrollment_test.csv', '')
    exit()

    BLENDER_PATH = os.path.dirname(os.path.abspath(__file__))
    print BLENDER_PATH

    if len(sys.argv) == 1:
        #os.chdir('./exp')
        os.chdir(TMPFS_PATH + '/exp')
        for i in range(1):
            result = subprocess.check_output(
                    './run_exps.sh ' + str(len(X[0])-i) + ' ' +\
                    BLENDER_PATH + '/blender.py',
                    #'/tmp2/b01902066/blending/blender.py',
                    shell=True
                )
            result = subprocess.check_output(
                    './view_result.sh ' + str(len(X[0])-i),
                    shell=True
                )
            res = parse_exp_result(result)
            #with open(BLENDER_PATH + '/blending.model', 'r') as f:
            #    w = f.readlines()[-1].split()[1:-1]
            ind = np.argsort(res)[::-1]
            #print res, w
            for i in ind:
                print (filelist[i], res[i])#, w[i].split(':')[1])
            print ''
            #print 'remove:', filelist[ind[0]]
            #print 'max auc:', res[ind[0]]
            #os.remove(PRED_PATH + filelist[ind[0]])
            del filelist[ind[0]]
        #os.chdir(TMPFS_PATH + 'blending/')
        os.chdir(os.path.dirname(__file__))

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


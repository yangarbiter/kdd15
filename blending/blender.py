
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

SVM_PERF_PATH = '../current/svm_perf/'

def perf_SVM_train_eval(X, y, valX, valy, C):
    train_file = 'blend' + str(C) + '.svmlight'
    model_file = 'blending' + str(C) + '.model'
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
            val_file + ' ' + model_file,
            shell=True
        )
    return float(re.findall('ROCArea[\s:0-9\.]+', result)[0][11:])/100.0


class Blender():
    def __init__(self):
        pass

    def grid_search(self, X, y, valX, valy):
        candidate_c = [10**i for i in range(-5, 5)]
        res = Parallel(n_jobs=10)(
                    delayed(perf_SVM_train_eval)(X, y, valX, valy, c) \
                        for c in candidate_c)
        print "grid search result: ", res
        return candidate_c[res.index(max(res))]

    def train(self, X, y, C=1.0, gamma=1.0):
        #parameters = {
        #            'kernel': ['rbf'],
        #            'C': [0.01, 0.1, 1, 10, 100],
        #            'gamma': [0.01, 0.1, 1, 10, 100]
        #        }
        #self.clf = GridSearchCV(
        #            SVC(kernel='rbf'),
        #            parameters,
        #            cv=5,
        #            n_jobs=-1
        #        )
        #self.clf = SVC(
        #            kernel='rbf',
        #            C=C,
        #            gamma=gamma,
        #            probability=True,
        #            cache_size=160000,
        #        )

        #self.clf = SVC(
        #        kernel='linear',
        #        C=10.0,
        #        class_weight="auto",
        #        probability=True,
        #    )
        #self.clf = RandomForestClassifier(
        #        n_estimators=500,
        #        class_weight="auto",
        #        n_jobs=-1,
        #    )
        #self.clf = LogisticRegression(
        #        C=0.1,
        #    )
        #self.clf.fit(X, y)
        #print self.clf.grid_scores_
        #print self.clf.coef_

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

def read_preds(path, exclude=-1):
    preds = []
    valpreds = []
    testpreds = []
    for i, filename in enumerate(os.listdir(path)):
        #if 'ada' in filename[:4]: continue
        #if 'RF' in filename: continue
        if 'team5' in filename: continue
        if i == exclude:
            print 'exclude: ' + filename
            continue
        #if 'DNN' in filename: continue
        #if 'LogR' in filename: continue
        if '_blend.csv' not in filename: continue

        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pred.append(float(row[1]))
        #print(len(pred))
        pred = calibrate(pred)
        preds.append(pred)
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
    x = np.array(x)
    order = x.argsort()
    ranks = order.argsort()
    return ranks / np.float(len(ranks))

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

def main():
    eid, y, valeid, valy = read_truth()
    X, valX, testX = read_preds('./allpreds/0618/')

    #print X, testX
    print np.shape(X), np.shape(valX), np.shape(testX)

    #scaler = MinMaxScaler()
    #_ = scaler.fit_transform(np.vstack((X, valX, testX)))
    #X = _[:len(X), :]
    #valX = _[len(X):len(X)+len(valX), :]
    #testX = _[len(X)+len(valX):, :]

    #X = np.vstack((X, testX))
    #y = np.hstack((y, testy))
    #result = []
    #kf = KFold(len(y), n_folds=40, shuffle=True)
    #for tri, tei in kf:
    #    clf = Blender()
    #    clf.train(X[tri], y[tri])
    #    result.append(clf.auc_score(X[tei], y[tei]))
    #print np.mean(result)
    #exit()

    clf = Blender()
    C = clf.grid_search(X, y, valX, valy)
    print C
    clf.ori_score(X, y)
    print "------------------"
    clf.ori_score(valX, valy)
    print "------------------"
    #for c in [0.001, 0.1, 1.0, 100]:
    #    for g in [0.001, 0.1, 1.0, 100]:
    #        clf = Blender()
    #        clf.train(X, y, c, g)
    #        print '%f %f' % (c, g)
    #        print 'auc in: ', clf.auc_score(X, y)
    #        print 'auc val:', clf.auc_score(valX, valy)
    #        print "------------------"
    #exit()
    clf.train(X, y, C)
    #clf.predict(X, y)
    print 'auc in: ', clf.predict(X, y)
    print 'auc val:', clf.predict(valX, valy)

    #print 'auc in: ', clf.auc_score(X, y)
    #print 'auc val:', clf.auc_score(valX, valy)

    dump_svmlight_file(testX, np.zeros((len(testX))), 'test.svmlight', zero_based=False)
    result = subprocess.check_output(
        '../current/svm_perf/svm_perf_classify test.svmlight ./blending.model ./blending_prediction',
        shell=True
    )
    outputans([], '/tmp2/b01902066/KDD/data/enrollment_test.csv', '')

    #print result
    #outputans(clf.predict(testX)[:, 1],
    #        '/tmp2/b01902066/KDD/data/enrollment_test.csv',
    #        '610_gbm_gbmrank_rf_ada_polylog_blend_logist_ans.csv')

    #print '3 uniform auc val:', roc_auc_score(valy, np.mean(valX[:, :3], axis=1))



if __name__ == "__main__":
    main()



import numpy as np
import os, csv, subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

class Blender():
    def __init__(self):
        pass

    def train(self, X, y):
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
        self.clf = LogisticRegression(
                C=0.1,
                class_weight='auto',
            )
        self.clf.fit(X, y)
        print self.clf.coef_
        #dump_svmlight_file(X, y, 'blend.svmlight', zero_based=False)
        #result = subprocess.check_output(
        #        '../current/svm_perf/svm_perf_learn -c 0.1 -l 10 -w 3 blend.svmlight blending.model',
        #        shell=True
        #    )

    def ori_score(self, X, y):
        for i in range(len(X[0])):
            print roc_auc_score(y, X[:, i])


    def predict(self, X, y=[]):
        #dump_svmlight_file(X, y, 'test.svmlight', zero_based=False)
        #result = subprocess.check_output(
        #        '../current/svm_perf/svm_perf_classify ./blending.model test.svmlight ./blending_prediction',
        #        shell=True
        #    )
        #print result
        return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict_proba(X)[:, 1])

def read_preds(path, testpath):
    preds = []
    testpreds = []
    for filename in os.listdir(path):
        #eid_prob = {}
        if 'rank' in filename[:4]:
            continue
        pred = []
        with open(path+filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                #eid_prob[int(row[0])] = float(row[1])
                pred.append(float(row[1]))
        #preds.append(eid_prob)
        preds.append(pred)

        print filename
        test_filename = filename[:-9] + 'test.csv'

        #eid_prob = {}
        pred = []
        with open(testpath+test_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                #eid_prob[int(row[0])] = float(row[1])
                pred.append(float(row[1]))
        #testpreds.append(eid_prob)
        testpreds.append(pred)

    return np.array(preds).T, np.array(testpreds).T

def read_truth():
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

    return eid, y, testeid, testy

def main():
    eid, y, testeid, testy = read_truth()
    X, testX = read_preds('./preds/', './testpreds/')

    #print X, testX
    print np.shape(X), np.shape(testX)

    scaler = MinMaxScaler()
    _ = scaler.fit_transform(np.vstack((X, testX)))
    X = _[:len(X), :]
    testX = _[len(X):, :]

    y = np.array(y)
    testy = np.array(testy)

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
    clf.ori_score(X, y)
    print "------------------"
    clf.ori_score(testX, testy)
    print "------------------"
    clf.train(X, y)
    clf.predict(testX, testy)
    print clf.auc_score(X, y)
    print clf.auc_score(testX, testy)


if __name__ == "__main__":
    main()


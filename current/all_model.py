
from generate_feature import *
import cPickle, csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from joblib import Parallel, delayed
from multiprocessing import Process

def loadans(filename):
    trainy = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            enrole_id = int(row[0])
            trainy.append(int(row[1]))
    return trainy

def outputans(ans, id_file_paths, path):
    idxs = []
    for id_file_path in id_file_paths:
        with open(id_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'enrollment_id': continue
                idxs.append(int(row[0]))
    assert len(idxs) == len(ans)

    with open(path, 'w') as f:
        for i, enrole_id in enumerate(idxs):
            f.write(str(enrole_id) + ',' + str(ans[i]) + '\n')

def poly2minmax(trainX, predX):
    poly = PolynomialFeatures(2)
    trainX = poly.fit_transform(trainX)
    predX = poly.fit_transform(predX)

    scaler = MinMaxScaler()
    trainX = scaler.fit_transform(trainX)
    predX = scaler.fit_transform(predX)
    return trainX, predX

def identity(trainX, predX):
    return trainX, predX

def train_pred(clf, trainX, y, predX, eidfile, dest):
    clf.fit(trainX, y[:len(trainX)])
    #print roc_auc_score(y[len(trainX):len(trainX)+len(predX)], clf.predict_proba(predX)[:, 1])
    outputans(clf.predict_proba(predX)[:, 1], eidfile, dest)

clfrfent = RandomForestClassifier(
        criterion='entropy',
        n_estimators = 3000,
        #class_weight = "auto",
        min_samples_split = 30,
        max_depth = 15,
        n_jobs=-1
    )

clfrfgini = RandomForestClassifier(
        criterion='gini',
        n_estimators = 3000,
        #class_weight = "auto",
        min_samples_split = 30,
        max_depth = 15,
        n_jobs=-1
    )

clflogis = LogisticRegression(
        C=0.1,
        dual=False,
    )

clfada=AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=3,
        ),
    learning_rate = 0.05,
    n_estimators = 600,
)

models = [
        {
            'mod': clfrfent,
            'mod_name': 'rfentro',
            'preproc': identity
        },
        {
            'mod': clfrfgini,
            'mod_name': 'rfgini',
            'preproc': identity
        },
        {
            'mod': clflogis,
            'mod_name': 'logis',
            'preproc': poly2minmax
        },
        {
            'mod': clfada,
            'mod_name': 'ada',
            'preproc': identity
        }
        ]


def main():
    DATAPATH_BASE = '/tmp2/b01902066/KDD/kdd15/current/data/0623/'
    BASE_PATH = '/tmp2/b01902066/KDD/kdd15/blending/allpreds/mypreds/'

    with open(DATAPATH_BASE + 'blend_train_feature.npy', 'r') as f:
        blendX = np.load(f)
    with open(DATAPATH_BASE + 'val_train_feature.npy', 'r') as f:
        valX = np.load(f)
    with open(DATAPATH_BASE + 'real_train_feature.npy', 'r') as f:
        testX = np.load(f)

    with open(DATAPATH_BASE + 'blend_test_feature.npy', 'r') as f:
        pred_blendX = np.load(f)
    with open(DATAPATH_BASE + 'val_test_feature.npy', 'r') as f:
        pred_valX = np.load(f)
    with open(DATAPATH_BASE + 'real_test_feature.npy', 'r') as f:
        pred_testX = np.load(f)

    #with open('/tmp2/b01902066/KDD/kdd15/current/data/0610/label_train+blend+valid.npy') as f:
    #    y = np.load(f)
    trainy = loadans('/tmp2/kdd/truth_train.csv')
    blendy = loadans('/tmp2/b01902066/KDD/data/internal1/truth_train.csv')
    valy = loadans('/tmp2/b01902066/KDD/data/truth_train.csv')

    processes = []

    for model in models:
        tX, pX = model['preproc'](blendX, pred_blendX)
        proc = Process( target=train_pred,
                args=(
                    model['mod'], tX, trainy, pX,
                    ['/tmp2/kdd/enrollment_blend.csv'],
                    BASE_PATH + model['mod_name'] + '_blend.csv',
                ))
        processes.append(proc)
        processes[-1].start()

        tX, pX = model['preproc'](valX, pred_valX)
        proc = Process( target=train_pred,
                args=(
                    model['mod'], tX, blendy, pX,
                    ['/tmp2/kdd/enrollment_val.csv'],
                    #'/tmp2/b01902066/KDD/data/internal1/enrollment_test.csv',
                    BASE_PATH + model['mod_name'] + '_val.csv',
                ))
        processes.append(proc)
        processes[-1].start()

        tX, pX = model['preproc'](testX, pred_testX)
        proc = Process( target=train_pred,
                args=(
                    model['mod'], tX, valy, pX,
                    ['/tmp2/b01902066/KDD/data/enrollment_test.csv'],
                    BASE_PATH + model['mod_name'] + '_test.csv',
                ))
        processes.append(proc)
        processes[-1].start()

    for i in processes:
        i.join()

if __name__ == '__main__':
    main()

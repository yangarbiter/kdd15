
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from blender import read_truth, read_preds, PRED_PATH
from sklearn.preprocessing import OneHotEncoder
import os, csv, subprocess, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

class nnBlender():
    def __init__(self, num_features):
        self.batch_size = 16

        self.model = Sequential()
        self.model.add(Dense(20, 64, init='uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, 64, init='uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, 2, init='uniform'))
        self.model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=sgd)

    def train(self, X, y, valX, valy):
        enc = OneHotEncoder()
        y = enc.fit_transform(np.reshape(y, (-1, 1)))
        valy = enc.fit_transform(np.reshape(valy, (-1, 1)))
        print np.shape(valy)
        self.model.fit(
                X, y,
                nb_epoch=20,
                batch_size=self.batch_size,
                show_accuracy=True,
                validation_data=(valX, valy))
        score = self.model.evaluate(valX, valy, batch_size=self.batch_size)
        return score

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def auc_score(self, X, y):
        return roc_auc_score(y, self.clf.predict(X)[:, 1])

def main():
    filelist = []
    removed = []
    for filename in sorted(os.listdir(PRED_PATH)):
        if '_blend.csv' in filename and filename not in removed:
            filelist.append(filename)

    eid, y, valeid, valy = read_truth()
    X, valX, testX, filelist = read_preds(PRED_PATH, filelist)

    print np.shape(X), np.shape(valX), np.shape(testX)

    clf = nnBlender(len(X[0]))
    print clf.train(X, y, valX, valy)
    print clf.predict(testX)


if __name__ == '__main__':
    main()


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
        self.batch_size = 200

        self.model = Sequential()
        self.model.add(Dense(num_features, 64, init='glorot_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(64, 2, init='glorot_normal'))
        self.model.add(Activation('softmax'))

        sgd = SGD(lr=0.0005, decay=1e-5, momentum=0.9, nesterov=True)
        #self.model.compile(loss='binary_crossentropy', optimizer=sgd)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def train(self, X, y, valX, valy):
        enc = OneHotEncoder()
        y = enc.fit_transform(np.reshape(y, (-1, 1)))
        valy = enc.fit_transform(np.reshape(valy, (-1, 1)))
        y = y.toarray()
        valy = valy.toarray()
        print np.shape(y)
        print type(y)
        print np.shape(valy)
        self.model.fit(
                X, y,
                nb_epoch=300,
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
    print roc_auc_score(valy, clf.predict(valX)[:, 1])
    print np.shape(clf.model.layers[0].get_weights()),
    print np.shape(clf.model.layers[0].get_weights()[0]),
    print np.shape(clf.model.layers[0].get_weights()[1])
    print clf.model.layers[0].get_weights()


if __name__ == '__main__':
    main()

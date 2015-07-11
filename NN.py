#from __future__ import absolute_import
import numpy as np
import sys
import theano
import csv
#from __future__ import print_function
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
#from keras.constraints import maxnorm
from sklearn.metrics import auc, roc_curve, roc_auc_score
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler as MMS
from keras.callbacks import ModelCheckpoint

batch_size = 256
n_classes = 2
n_epoch = 100
#lr = 0.01

np.random.seed(1337) # for reproducibility


# convert class vectors to binary class matrices


model = Sequential()

model.add(Dense(95, 60, init='glorot_uniform') )
model.add(PReLU(60))
#model.add(Dropout(0.01))
model.add(Dense(60, 30, init='glorot_uniform'))
model.add(PReLU(30))
#model.add(Dropout(0.01))
model.add(Dense(30, 10, init='glorot_uniform'))
model.add(PReLU(10))

#model.add(Dense(512, 256, init='glorot_uniform'))
#model.add(Activation('relu'))
#model.add(PReLU(256))
#model.add(Dropout(0.01))

#model.add(Dense(256, 256, init='glorot_uniform'))
#model.add(Activation('relu'))
#model.add(PReLU(256))
#model.add(Dropout(0.01))

#model.add(Dense(256, 32, init='glorot_uniform'))
#model.add(Activation('relu'))
#model.add(PReLU(32))
#model.add(Dropout(0.01))


model.add(Dense(10, 1, init='glorot_uniform' ))
model.add(Activation('sigmoid'))

sgd = SGD(lr = 0.1,momentum=0.5)
rms = RMSprop()
#model.compile(loss='binary_crossentropy', optimizer=rms)
model.compile(loss='pairwise_loss', optimizer=rms)

PREFIX = "../data/"
#load data
trainX = np.load(open(PREFIX+"train_feature.npy", "rb"))
blendX = np.load(open(PREFIX+"blend_feature.npy", "rb"))
validX = np.load(open(PREFIX+"valid_feature.npy", "rb"))
testX = np.load(open(PREFIX+"test_feature.npy", "rb"))
trainX = np.vstack((trainX,blendX))

Y = np.load(open(PREFIX+"label_train+blend+valid.npy", "rb"))
evalY = Y[len(trainX):]

#Y = np_utils.to_categorical(Y, n_classes)
trainY = Y[:len(trainX)]
validY = Y[len(trainX):]

#scaling
split1 = len(trainX)
split2 = len(trainX)+len(validX)
X = np.vstack((trainX, validX))
X = np.vstack((X, testX))
X = MMS().fit_transform(X)
trainX = X[:split1]
validX = X[split1:split2]
testX = X[split2:]


print trainX.shape
print validX.shape
print testX.shape


checkpointer = ModelCheckpoint(filepath = "../train+blend.mod.hdf5", save_best_only=True, verbose=2)
model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=n_epoch, show_accuracy=True, verbose=1, validation_data = (validX, validY), callbacks=[checkpointer])
model.load_weights("../train+blend.mod.hdf5")
model2 = Sequential()

model2.add(Dense(95, 60, weights = model.layers[0].get_weights()))
model2.add(PReLU(60))

model2.add(Dense(60, 30, weights = model.layers[2].get_weights()))
model2.add(PReLU(30))


model2.add(Dense(30, 10, weights = model.layers[4].get_weights()))
model2.add(PReLU(10))
model2.compile(loss='binary_crossentropy', optimizer=rms)
#model2.compile(loss='pairwise_loss', optimizer=rms)
X = model2.predict(trainX)
X.dump("train+blend_95_NN_feature.npy")
X = model2.predict(validX)
X.dump("valid_95_NN_feature.npy")
X = model2.predict(testX)
X.dump("test_95_NN_feature.npy")
predY = model.predict_proba(validX)
print roc_auc_score(evalY, predY) 
f = open(PREFIX+"team2_DNN95_val.csv", "w")
for (line, prob) in zip(open(PREFIX+"valid_prediction.csv", "r"), predY):
    ID = line.strip().split(",")[0]
    f.write(ID+","+str(prob)+"\n")

predY = model.predict_proba(testX)[:,1]
f = open(PREFIX+"team2_DNN95_test.csv", "w")
for (line, prob) in zip(open(PREFIX+"test_prediction.csv", "r"), predY):
    ID = line.strip().split(",")[0]
    f.write(ID+","+str(prob)+"\n")

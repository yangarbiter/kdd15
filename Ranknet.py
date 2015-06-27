"""include modules"""
import os
import sys
import time
import cPickle
import numpy as np
import random
import csv
import theano
import theano.tensor as T
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler as MM
from sklearn.metrics import auc, roc_curve, roc_auc_score
from numpy.linalg import norm

"""theano config settings"""
theano.config.openmp=True
#theano.config.nvcc.fastmath=True
theano.config.floatX = 'float32'

def relu(x):
    return  x * (x > 0)

def to_pairwise(X, Y, K):
    SP = []
    SN = []
    for i in range(Y.shape[0]):
        if Y[i] == 0:
            SN.append(i)
        else:
            SP.append(i)
    X1 = []
    X2 = []
    Y_out = []
    for i in range(X.shape[0]):
        for k in range(K):
            X1.append(X[i])
            Y_out.append(Y[i])
            if Y[i] == 1:
                #j = random.choice(SN)
                sn = random.sample(SN, 20)
                D = {j:norm(X[i] - X[j]) for j in sn}
                j = min(D, key =D.get)
                X2.append(X[j])
            else:
                #j = random.choice(SP)
                sp = random.sample(SP, 20)
                D = {j:norm(X[i] - X[j]) for j in sp}
                j = min(D, key =D.get)
                X2.append(X[j])
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y_out = np.array(Y_out)
    return (X1, X2, Y_out)
"""
load the dataset that is in svmlight format,
if the valid is Ture, it will split 20% from the dataset for validation
"""
def load_my_data(dataset, valid = True, test = False):
    print '... loading data :', dataset

    #laod X
    with open(sys.argv[1], "rb") as f:
        #X = cPickle.load(f)
        X = np.load(f)
    print "the whole shape X", X.shape
    if test is True:
        with open(sys.argv[2], "rb") as f:
            #testX = cPickle.load(f)
            testX = np.load(f)
        print "the whole shape testX", testX.shape
        N = len(X)
        X = np.vstack((X, testX))
        X = MM().fit_transform(X)
        testX = X[N:]
        X = X[:N]
    else:
        X = MM().fit_transform(X)

    #load Y
    Y = []
    with open(sys.argv[3], 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            Y.append(int(row[1]))
    Y = np.array(Y)

    #shuffle
    order = [i for i in range(len(Y))]
    np.random.shuffle(order)
    X_new = np.array([X[i] for i in order])
    Y_new = np.array([Y[i] for i in order])
    X = X_new
    Y = Y_new
    
    if valid == True:# 20% for validation
    	    split = int(len(Y)*0.8)+1
    else:
	    split = len(Y)
    
    trainX = X[:split]
    trainY = Y[:split]
    validX = X[split:]
    validY = Y[split:]
    
    print "to pairwise"
    (trainX1, trainX2, trainY)  = to_pairwise(trainX, trainY, 1)
    (validX1, validX2, validY)  = to_pairwise(validX, validY, 1)
    print trainX1.shape, trainX2.shape, trainY.shape
    print validX1.shape, validX2.shape, validY.shape
    
    
    """make the data into theano shared variables"""
    def shared_dataset(data_x1, data_x2, data_y, borrow=True):
        shared_x1 = theano.shared(np.asarray(data_x1,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_x2 = theano.shared(np.asarray(data_x2,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x1, shared_x2, T.cast(shared_y, 'int32')
    
    def shared_test_dataset(data_x, borrow=True):
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x
    valid_set_x1, valid_set_x2,valid_set_y = shared_dataset(validX1, validX2, validY)
    train_set_x1, train_set_x2,train_set_y = shared_dataset(trainX1, trainX2, trainY)
    

    if test is True:
        test_set_x = shared_test_dataset(testX)
        rval = [(train_set_x1, train_set_x2, train_set_y), (valid_set_x1, valid_set_x2, valid_set_y), (test_set_x)]
    else:
        rval = [(train_set_x1, train_set_x2, train_set_y), (valid_set_x1, valid_set_x2, valid_set_y)]
    return rval

"""
This is the otput layer of the model.
We use the negative log likelihood as the loss function.
The error fuction used for validation are 0/1 error.
The probability of each label y is of softmax distribution.
"""
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                        dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=np.zeros((n_out, ),
                                        dtype=theano.config.floatX), name='b')
        x1 = T.dot(input[0], self.W) + self.b
        self.p_y_given_x1 = T.exp(x1)/T.exp(x1).sum(1, keepdims = True)#softmax distribution
        x2 = T.dot(input[1], self.W) + self.b
        self.p_y_given_x2 = T.exp(x2)/T.exp(x2).sum(1, keepdims = True)#softmax distribution
        p = x1 - x2
        #p = self.p_y_given_x1 - self.p_y_given_x2
        self.P = T.nnet.sigmoid(p)[:,1]
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
    #     return -T.mean(T.log(self.P)[T.arange(y.shape[0]), y])
        return T.mean(T.nnet.binary_crossentropy(self.P, y))

"""
In this class we define a hiddenlayer.
Each has its input, output, random stream(rng), W, b, and the activation function.
We use tanh as out activation function.
"""
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        
        self.input = input
        if W is None:#magic initialized values of W
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = (T.dot(input[0], self.W) + self.b , T.dot(input[1], self.W) + self.b)
        
        self.output = (

            lin_output if activation is None
            else activation(lin_output[0]), activation(lin_output[1])
        )
        self.params = [self.W, self.b]


"""
The main class of our Dnn which performs training and predictition.
"""
class Dnn(object):
    def __init__(self, rng, n_in = 28*28, n_hidden = [500], n_out = 10):
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x1 = T.matrix('x1')  # the data is presented as a feature matriix
        self.x2 = T.matrix('x2')  # the data is presented as a feature matriix
        self.y = T.ivector('y')  # the labels are presented as 1D vector of

        self.hiddenLayers = []
        self.params = []

        next_input = (self.x1, self.x2)
        next_n_in = n_in
        
        #more hidden layers
        for n_h in n_hidden:
            h = HiddenLayer(
            rng=rng,
            input=next_input,
            n_in=next_n_in,
            n_out=n_h,
            activation=T.tanh
            )
            next_input = h.output
            next_n_in= n_h
            self.hiddenLayers.append(h)
            self.params += h.params

        #output layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )
        self.params += self.logRegressionLayer.params

        #L1 regularizer
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for h in self.hiddenLayers:
            self.L1 += abs(h.W).sum()

        #L1 regularizer
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
        for h in self.hiddenLayers:
            self.L2_sqr += (h.W ** 2).sum()
        
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        #self.errors = self.logRegressionLayer.errors
    
    """perfrom training and prediction(if specified)"""
    def train(self, learning_rate=0.01, L1_reg=0.000, L2_reg=0.0000, n_epochs=2000,
                 dataset="", batch_size=128, test = False):

        #===========================load data===================================
        
        
        datasets = load_my_data(dataset, test = test)
        train_set_x1, train_set_x2, train_set_y = datasets[0]
        valid_set_x1, valid_set_x2, valid_set_y = datasets[1]
        if test is True:
            test_set_x = datasets[2]
            test_f = open(sys.argv[4], "w")
        

        n_train_batches =int( np.ceil(float(train_set_x1.get_value(borrow=True).shape[0]) / batch_size))
        n_valid_batches = valid_set_x1.get_value(borrow=True).shape[0] / batch_size


        #===========================build model================================
        print '... building the model'
        index = self.index
        x1 = self.x1
        x2 = self.x2
        y = self.y

        #the cost function we gradient on it
        cost = (
            self.negative_log_likelihood(y)
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )

        #gradient 
        gparams = [T.grad(cost, param) for param in self.params]

        #geadient descent
        updates = [
            (param, param - learning_rate * gparam )
            for param, gparam in zip(self.params, gparams)
        ]

        #theano training function
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x1: train_set_x1[index * batch_size: (index + 1) * batch_size],
                x2: train_set_x2[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        #theano validation function
        validate_model = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x1: valid_set_x1[index * batch_size:(index + 1) * batch_size],
                x2: valid_set_x2[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
        #get_train_pred = theano.function(inputs = [], 
        #        outputs=self.logRegressionLayer.p_y_given_x1,
        #        givens={x1: train_set_x1})
        get_valid_pred = theano.function(inputs = [], 
                outputs=self.logRegressionLayer.p_y_given_x1,
                givens={x1: valid_set_x1})
        if test is True:
            get_test_pred = theano.function(inputs = [], 
                outputs=self.logRegressionLayer.p_y_given_x1,
                givens={x1: test_set_x})

        #===========================training================================
        print '... training'

        #early stop parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 10  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant


        validation_frequency = min(n_train_batches, patience / 2)

        epoch = 0
        best_score = -np.inf
        p_cnt = 0
        while (epoch < n_epochs) :
            epoch = epoch + 1
            #mini batch updates
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index# iteration number
                if (iter + 1) % validation_frequency == 0:
                    # compute 0/1 loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print "epoch:", epoch, "index", minibatch_index,
                    print "in sample loss:", minibatch_avg_cost
                    P = get_valid_pred()[:,1]
                    print "AUC Valid:", roc_auc_score(valid_set_y.eval(), P)
                    score = roc_auc_score(valid_set_y.eval(), P) 
                    if score > best_score:
                        best_score = score
                        p_cnt = 0
                    else:
                        p_cnt += 1
                    if p_cnt >= 7:
                        break
                    print score, best_score, p_cnt
            if p_cnt >= 7:
                break
        if test is True:
            P = get_test_pred()[:,1]
            for p in P:
                test_f.write(str(p)+"\n")
if __name__ == '__main__':
    
    #create Deep Neural Network object
    dnn = Dnn(
        rng=np.random.RandomState(1234),
        n_in= 66,
        n_hidden=[50, 40, 30, 20, 10],
        n_out= 2
    )

    """
    If the test_file and the output_file are given,
    it will make prediction and output to the file
    """
    #dnn.train(dataset = "../data/train.dat", test_file = "../data/test.dat", output_file = "../result/4_layer.tmp")

    dnn.train(dataset = sys.argv[1], n_epochs = 2000, learning_rate = 0.01, test = True)

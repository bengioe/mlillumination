import struct
import numpy
import theano
import theano.tensor as T
from math import*
rng = numpy.random.RandomState(42)

hard_rectifier = lambda x: T.minimum(1,T.maximum(0,x))
soft_rectifier = lambda x: 1.0/(1+T.exp(-x))**2
rectifier = lambda x: T.maximum(0,x)

class HiddenLayer:
    def __init__(self,x,n_in,n_out,activation = T.tanh):
        self.x = x
        k = sqrt(6.0/(n_in+n_out))
        W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
        print (n_in,n_out)
        print W_vals[0][0], W_vals[0][1], W_vals[1][0]
        self.W = theano.shared(value=W_vals,name='W')
        self.b = theano.shared(0.25*numpy.ones((n_out,)),name='b')
        self.output = activation(T.dot(self.x,self.W)+self.b)
        self.params = [self.W,self.b]

class Model:
    def __init__(self, x, n_hidden=30, n_hidden2=30):
        self.layerA = HiddenLayer(x, 12, n_hidden, rectifier)
        self.layerB = HiddenLayer(self.layerA.output, n_hidden, n_hidden2, rectifier)
        #self.layerC = HiddenLayer(self.layerB.output, n_hidden2, 3, soft_rectifier)#T.nnet.sigmoid)
        self.layerC = HiddenLayer(self.layerB.output, n_hidden2, 3, T.nnet.sigmoid)
        self.params = self.layerA.params + self.layerB.params + self.layerC.params

        self.output = self.layerC.output

def dump_weigths(W, f):
    i = lambda x: struct.pack('i',x)
    d = lambda x: struct.pack('f',x)
    f.write(i(len(W)))
    
    for w in W:
        print w.get_value().shape
        for dim in w.get_value().shape:
            print dim
            f.write(i(dim))
        print w.get_value().flatten().shape
        for val in w.get_value().flatten():
            f.write(d(val))

#target="527.996, 398.541, -759.032" 

def train_model(train, test):
    
    LR = .1
    tau = 100
    batch_size = 8
    nepochs = 100
    y_epsilon = 1e-2
    nbatches = train[0].shape[0] / batch_size
    print nbatches, "batches"
    lr = T.scalar()
    x = T.matrix()
    y = T.matrix()

    model = Model(x, n_hidden=20, n_hidden2=10)

    cost = T.mean(abs(y-model.output)/(y+y_epsilon))
    rgbcost = T.mean(abs(y-model.output))
    
    gradients = T.grad(cost, model.params)
    updates = []
    for p,g in zip(model.params, gradients):
        updates.append((p, p - lr*g))

    train_model = theano.function([x,y,lr],
                                  cost,
                                  updates = updates)
    test_model = theano.function([x,y],
                                  [cost,rgbcost])
    eval_model = theano.function([x],
                                 [model.layerA.output,
                                  model.layerB.output,
                                  model.output])

    t0 = time.time()
    ntotal = nepochs*nbatches
    ndone = 0
    for epoch in range(nepochs):
        cost = 0
        ilr = LR*(tau/(tau+epoch*1.0))
        for i in range(nbatches):
            cost += train_model(train[0][i*batch_size:(i+1)*batch_size],
                                train[1][i*batch_size:(i+1)*batch_size],
                                ilr)
            ndone += 1
            if i%50 == 0:
                timeElapsed = time.time()-t0
                timeRemaining = (ntotal-ndone)*timeElapsed/ndone
                print "                     \r",i,int(timeRemaining/3600),":",int(timeRemaining/60)%60,":",int(timeRemaining)%60,
        print " "*80,"\r",
        testcost, rgbcost = test_model(test[0], test[1])
        print epoch,cost/(nbatches*batch_size), testcost, rgbcost, ilr

    print eval_model([[1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0]])
    return model

data = file('dataset.dat','r').read()
data = numpy.fromstring(data, dtype='float32')
nsamples = data.shape[0]/15
data = data.reshape((nsamples,5,3))

data[:,1] /= 600
data[:,4] /= 600

print data.mean(), data.max(), data.min()
ntrain = int(nsamples * 0.8)
ntest = nsamples - ntrain
train = data[:ntrain]

numpy.random.seed(9001)
numpy.random.shuffle(train)

train = train[:,1:].reshape((ntrain,3*4)),train[:,0].reshape((ntrain, 3))
test = data[ntrain:,1:].reshape((ntest,3*4)),data[ntrain:,0].reshape((ntest, 3))

import time

if 1:

    model = train_model(train, test)
    dump_weigths(model.params, file('weights'+time.strftime("%m.%d_%H.%M.%S")+'.dat','w'))
    #dump_weigths(model.params, file('weights_40.dat','w'))

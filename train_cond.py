import struct
import numpy
import theano
import theano.tensor as T
from math import*

theano.config.compute_test_value = 'raise'

rng = numpy.random.RandomState(42)

trng = T.shared_randomstreams.RandomStreams(42)

#rectifier = lambda x: T.minimum(1,T.maximum(-1,x))
rectifier = lambda x: T.maximum(0,x)


def load_weights(f,skipndims=False):
    i = lambda : struct.unpack('i',f.read(4))[0]
    d = lambda : struct.unpack('f',f.read(4))[0]
    W = []
    n = i()
    for wi in range(n):
        if skipndims:
            ndims = [2,1,2,1,2,1][wi]
        else:
            ndims = i()
        dims = []
        flat_size = 1
        for di in range(ndims):
            dims.append(i())
            flat_size *= dims[-1]
        vals = []
        for vi in range(flat_size):
            vals.append(d())
        print ndims,dims
        W.append(numpy.array(vals).reshape(dims))
    return W

def dump_weigths(W, f):
    i = lambda x: struct.pack('i',x)
    d = lambda x: struct.pack('f',x)
    f.write(i(len(W)))
    
    for w in W:
        #print w.get_value().shape
        f.write(i(len(w.get_value().shape)))
        for dim in w.get_value().shape:
            #print dim
            f.write(i(dim))
        #print w.get_value().flatten().shape
        for val in w.get_value().flatten():
            f.write(d(val))


class HiddenLayer:
    def __init__(self,x,n_in,n_out,activation = T.tanh):
        self.x = x
        k = sqrt(6.0/(n_in+n_out))
        W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
        #print (n_in,n_out)
        self.W = theano.shared(value=W_vals,name='W')
        self.b = theano.shared(0.25*numpy.ones((n_out,)),name='b')
        self.output = activation(T.dot(self.x,self.W)+self.b)
        self.params = [self.W,self.b]

class RecHiddenLayer:
    def __init__(self,x,n_in,n_out,activation = T.tanh):
        self.x = x
        k = sqrt(6.0/(n_in+n_out))
        W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
        #print (n_in,n_out)
        self.W = theano.shared(value=W_vals,name='Wrec')
        self.b = theano.shared(numpy.zeros((n_out,)),name='brec')
        self.bp = theano.shared(numpy.zeros((n_in,)),name='bp')
        self.output = activation(T.dot(self.x,self.W)+self.b)
        self.params = [self.W, self.b]
        self.rparams = [self.W, self.b, self.bp]
        
    def setrec(self, y, activation=T.tanh):
        self.rec = activation(T.dot(y,self.W.T)+self.bp)


_qw =  load_weights(file("W_weights_0/weights12.09_21.44.45.dat",'r'),True)

class RGBModel:
    def __init__(self, x, n_hidden=30, n_hidden2=30):
        self.layerA = HiddenLayer(x, 12, n_hidden, rectifier)
        self.layerB = HiddenLayer(self.layerA.output, n_hidden, n_hidden2, rectifier)
        self.layerC = HiddenLayer(self.layerB.output, n_hidden2, 3, T.nnet.sigmoid)
        self.params = self.layerA.params + self.layerB.params + self.layerC.params
        #for i,pi in zip(_qw, self.params):
        #    pi.set_value(i)

        self.output = self.layerC.output

class SelectorLayers:
    def __init__(self, x, n_hidden=100, n_desc=6):
        self.layerA = RecHiddenLayer(x, 12, n_hidden, T.tanh)
        self.layerB = RecHiddenLayer(self.layerA.output, n_hidden, n_hidden,  T.tanh)
        self.layerC = RecHiddenLayer(self.layerB.output, n_hidden, n_desc,  T.nnet.sigmoid)
        #lambda x: T.nnet.softmax(T.nnet.sigmoid(x)*30))
        self.layerC.setrec(self.layerC.output)
        self.layerB.setrec(self.layerC.rec)
        self.layerA.setrec(self.layerB.rec)
        self.rec = self.layerA.rec
        self.params = self.layerA.params + self.layerB.params + self.layerC.params
        self.rparams = self.layerA.rparams + self.layerB.rparams + self.layerC.rparams
        self.rcost = T.mean((x-self.rec)**2) + 0.00001*sum([T.sum(i**2) for i in self.params[::2]])
        self.rcost += 6*T.var(T.mean(self.layerC.output, axis=0))
        self.probs = self.layerC.output
        self.output = self.probs

class SelectorModel:
    def __init__(self, x, y, smodel, n_subs, model):
        self.subs = []
        self.params = []
        self.selector = smodel
        self.ys = []

        probs = self.selector.output
        pbs = T.argmax(probs,axis=1)
        cost = 0
        rgbcost = 0
        for i in range(n_subs):
            eq = T.concatenate([T.as_tensor_variable(numpy.array([0])),T.eq(pbs,i).nonzero()[0]])
            subx = x[eq]
            suby = y[eq]
            self.subs.append(model(subx))
            self.params += self.subs[-1].params
            self.ys.append(self.subs[-1].output)
            rgbcost += T.mean(abs(self.subs[-1].output-suby)) / n_subs
            cost += T.mean(abs(self.subs[-1].output-suby)/(suby+1e-2)) / n_subs

        self.export_params = self.params + self.selector.params
        
        #self.selector.output = T.printing.Print('dprobs',("__str__",))(self.selector.output)

        # ys = (n_subs,batchsize,3)
        ys = T.concatenate(self.ys)
        self.probs = probs
        self.cost = cost
        self.cost += 0.000001 * T.mean(T.as_tensor_variable(
                [T.sum(i**2) for i in self.params if 'W' in i.name]))
        
        #sout = self.selector.output
        #sout = T.printing.Print('sout',("shape",))(sout)
        #k = T.argmax(sout,axis=1)
        #self.k = k
        #k = T.printing.Print('k',("shape","dtype"))(k)
        #ys = T.printing.Print('ys',("shape",))(ys)
        #self.output = ys[k,T.arange(x.shape[0])]
        self.rgbcost = rgbcost #T.mean(abs(self.output-y))

    def load(self, W):
        for w,sw in zip(W[:-4],self.params[:-4]):
            sw.set_value(w)

#target="527.996, 398.541, -759.032" 

def train_model(train, test):
    
    LR = 1
    tau = 30
    batch_size = 16
    nepochs = 10
    n_subs = 10
    nbatches = train[0].shape[0] / batch_size
    print nbatches, "batches"
    lr = T.scalar()
    lr.tag.test_value = 0.1
    x = T.matrix()
    x.tag.test_value = numpy.ones((17,12),dtype='float32')
    y = T.matrix()
    y.tag.test_value = numpy.ones((17,3),dtype='float32')

    smodel = SelectorLayers(x, 20, n_subs)
    cost = smodel.rcost
    gradients = T.grad(cost, smodel.rparams)
    updates = []
    for p,g in zip(smodel.rparams, gradients):
        updates.append((p, p - lr*g))

    train_model = theano.function([x,lr],
                                  cost,
                                  updates = updates)
    test_model = theano.function([x],
                                 [cost, smodel.probs])
    


    t0 = time.time()
    ntotal = nepochs*nbatches
    ndone = 0
    for epoch in range(nepochs):
        cost = 0
        ilr = LR*(tau/(tau+epoch*1.0))
        for i in range(nbatches):
            cost += train_model(train[0][i*batch_size:(i+1)*batch_size],
                                ilr)
            ndone += 1
            if i%1000 == 0:
                timeElapsed = time.time()-t0
                timeRemaining = (ntotal-ndone)*timeElapsed/ndone
                print "                     \r",i,int(timeRemaining/3600),":",int(timeRemaining/60)%60,":",int(timeRemaining)%60,
                sys.stdout.flush()
        print " "*80,"\r",
        terr, probs = test_model(train[0])
        print numpy.bincount(probs.argmax(axis=1))
        terr, probs = test_model(test[0])
        print numpy.bincount(probs.argmax(axis=1))
        print epoch,cost/(nbatches*batch_size), terr, ilr

    
    print "Building model..."
    model = SelectorModel(x, y, smodel, n_subs, lambda X: RGBModel(X,20,10))

    if 0:
        W = load_weights(file("weights12.15_14.48.31.dat",'r'))
        model.load(W)

    print "Computing gradient..."
    cost = model.cost
    
    gradients = T.grad(cost, model.params)
    updates = []
    for p,g in zip(model.params, gradients):
        updates.append((p, p - lr*g))

    print "Compiling theano functions..."
    train_model = theano.function([x,y,lr],
                                  cost,
                                  updates = updates)
    test_model = theano.function([x,y],
                                  [cost,model.rgbcost, model.probs])
    #eval_model = theano.function([x],
    #                             [model.output])



    LR = .5
    tau = 50
    nepochs = 20
    print "Training..."
    t0 = time.time()
    ntotal = nepochs*nbatches
    ndone = 0
    for epoch in range(nepochs):
        cost = 0
        ilr = LR*(tau/(tau+epoch*1.0))
        for i in range(nbatches):
            c = train_model(train[0][i*batch_size:(i+1)*batch_size],
                            train[1][i*batch_size:(i+1)*batch_size],
                            ilr)
            cost += c
            ndone += 1
            if i%1000 == 0:
                timeElapsed = time.time()-t0
                timeRemaining = (ntotal-ndone)*timeElapsed/ndone
                print "                     \r",i,int(timeRemaining/3600),":",int(timeRemaining/60)%60,":",int(timeRemaining)%60,cost/(i+1),c,
                sys.stdout.flush()
        print " "*80,"\r",
        terr, rgbcost, probs = test_model(test[0], test[1])
        #print numpy.bincount(probs.argmax(axis=1))
        print epoch,cost/(nbatches*batch_size), terr, rgbcost, ilr

    #print eval_model([[1,1,1,1,1,1,1,1,1,1,1,1],
    #                  [0,0,0,0,0,0,0,0,0,0,0,0]])
    return model

import sys
data = file('dataset.dat','r').read()
data = numpy.fromstring(data, dtype='float32')
nsamples = data.shape[0]/15
data = data.reshape((nsamples,5,3))

data[:,1] /= 600
data[:,4] /= 600

print data.mean(), data.max(), data.min()
ntrain = int(nsamples * 0.9)
ntest = nsamples - ntrain
train = data[:ntrain]

numpy.random.seed(9001)
numpy.random.shuffle(train)

train = train[:,1:].reshape((ntrain,3*4)),train[:,0].reshape((ntrain, 3))
test = data[ntrain:,1:].reshape((ntest,3*4)),data[ntrain:,0].reshape((ntest, 3))

import time

if 1:
    
    model = train_model(train, test)
    dump_weigths(model.export_params, file('weights'+time.strftime("%m.%d_%H.%M.%S")+'.dat','w'))
    #dump_weigths(model.params, file('weights_40.dat','w'))

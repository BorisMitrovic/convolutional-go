#!/usr/bin/env python
"""
My network 5x5 GO
"""

import cPickle
import gzip
import os
import sys
import time
import pickle

import numpy, random

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
import utils
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
import readGame


gs = 9 # size of the go board # TODO: IMPORTANT TO CHANGE!

def train(learning_rate=0.1, n_epochs=100, batch_size=320, batch_type = 'fast',
                    mynet = 'one', representation='raw', momentum=0, history=0):

    rng = numpy.random.RandomState(42)

    trainP = 0.8
    validP = 0.1
    testP  = 0.1   
    
#    print "... Reading cached values ..."
#    (trainCumLengths,validCumLengths,testCumLengths,filenames) = pickle.load(open("results/5x5.cache",'r'))
    
    print "... Getting filenames ..."
    datasetMY = "../MC player/20kgames9"
    fn1 = readGame.getFilenames(datasetMY,1,0,1)[0]
    random.shuffle(fn1)    
    filenames = fn1
    n = len(filenames)
    print "... Learning set contains " + str(n) + " games"
    
    print "... Computing cumulative game lengths ..."
    trainNames = filenames[:int(trainP*n)]
    validNames = filenames[int(trainP*n):int(trainP*n+validP*n)]
    testNames  = filenames[int(trainP*n+validP*n):int(trainP*n+validP*n+testP*n)]
    
    random.shuffle(trainNames)
    
    trainCumLengths = readGame.getCumGameLengths(trainNames,ftype="game")
    validCumLengths = readGame.getCumGameLengths(validNames,ftype="game")
    testCumLengths = readGame.getCumGameLengths(testNames,ftype="game")
    
    fw = open("results/"+str(gs)+"x"+str(gs)+".cache","wb")
    pickle.dump((trainCumLengths,validCumLengths,testCumLengths,filenames),fw)
    fw.close()
    print "... Preprocessing initial batches ..."
    minn = batch_size / 10 +1
    temp = time.time()
    test_batch_x, test_batch_y = utils.shared_dataset(readGame.processGAMEs(testNames[:minn],representation,gs=gs),batch_size=batch_size,board_size=gs)
    train_batch_x, train_batch_y = utils.shared_dataset(readGame.processGAMEs(trainNames[:minn],representation,gs=gs),batch_size=batch_size,board_size=gs)
    valid_batch_x, valid_batch_y = utils.shared_dataset(readGame.processGAMEs(validNames[:minn],representation,gs=gs),batch_size=batch_size,board_size=gs)
    print "    average processing time per game: " + str((time.time()-temp)/18.0) + " seconds, per epoch: " + str(int((time.time()-temp)/18*n/60/60)) + " hours" 

    # compute number of minibatches for training, validation and testing
    n_train_batches = trainCumLengths[-1]
    n_valid_batches = validCumLengths[-1]
    n_test_batches =  testCumLengths[-1]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    iteration = T.lscalar()  # iteration number of a minibatch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (gs, gs)  # this is the size of MNIST images

    fw = open("results/"+mynet+"_"+str(learning_rate)+"_"+".res","w")
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... Building the model ...'
   
    nc = 2 if representation=='raw' else 6  # if raw
    nc *= 1+history



       
    if mynet == "zero":
        layer0_input = x.reshape((batch_size, nc, gs, gs))
        layer0 = LogisticRegression(input=layer0_input.flatten(2), n_in=nc*gs*gs, n_out=gs*gs)
        cost = layer0.negative_log_likelihood(y)
    
        params = layer0.params

    if mynet == "one":
        nHiddens = 500
        layer1_input = x.reshape((batch_size, nc, gs, gs))
        layer1 = HiddenLayer(rng, input=layer1_input.flatten(2), n_in=nc * gs * gs,
                           n_out=nHiddens, activation=T.tanh)
        layer0 = LogisticRegression(input=layer1.output, n_in=nHiddens, n_out=gs*gs)
        cost = layer0.negative_log_likelihood(y)
    
        params = layer0.params + layer1.params
        
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([], layer0.errors(y),
             givens={
                x: test_batch_x,
                y: T.cast(test_batch_y, 'int32')})

    validate_model = theano.function([], layer0.errors(y),
             givens={
                x: valid_batch_x,
                y: T.cast(valid_batch_y, 'int32')})

    predictions = theano.function([], layer0.get_predictions(),
            givens={
                x: valid_batch_x})
                
    conditional_dist = theano.function([], layer0.get_conditional_dist(),
            givens={
                x: valid_batch_x})

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    #adjusted_rate = learning_rate - iteration*(learning_rate/(float(n_epochs) * n_train_batches))
    adjusted_rate = learning_rate if T.lt(iteration,3000*200) else 0.1*learning_rate
    
    for param_i, grad_i in zip(params, grads):#, prev_grad_i   , prevGrads):
        updates.append((param_i, param_i - adjusted_rate * grad_i))# - momentum * prev_grad_i))
    
    #for i,grad in enumerate(grads):
    #    updates.append((prevGrads[i], grad))
    
    train_model = theano.function([iteration], cost, updates=updates,
         givens={
            x: train_batch_x,
            y: T.cast(train_batch_y, 'int32')},on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... Training ...'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 2000         # min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    stime = time.time()

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 500 == 0:
                print 'training @ iter = ', iter
                pickle.dump((updates,cost,layer0,test_model,predictions,conditional_dist),open("results/"+str(batch_size)+representation+str(history)+".model","w"))
            if iter ==5:
                print 'estimated train time per epoch = '+ str((time.time() - stime) * n_train_batches/60.0/iter/60.0) + " hours"
            ax,ay = getBatch(trainNames, minibatch_index, trainCumLengths, batch_size,representation,batchType=batch_type,history=history)
            train_batch_x.set_value(ax)
            train_batch_y.set_value(ay)
            cost_ij = train_model(iter)

            if (iter + 1) % validation_frequency == 0 or iter==5:

                # compute zero-one loss on validation set
                validation_losses = []
                for i in xrange(n_valid_batches):
                    vx,vy = getBatch(validNames, i, validCumLengths, batch_size,representation,batchType='fast',history=history)
                    valid_batch_x.set_value(vx)
                    valid_batch_y.set_value(vy)
                    validation_losses.append(validate_model())
                this_validation_loss = numpy.mean(validation_losses)
        
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses=[]
                    for i in xrange(n_test_batches):
                        tx,ty = getBatch(testNames, i, testCumLengths, batch_size,representation,batchType='fast',history=history)
                        test_batch_x.set_value(tx)
                        test_batch_y.set_value(ty)
                        test_losses.append(test_model())
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

        #fw.write("Epoch "+str(epoch) + ": " +str((1-this_validation_loss)*100.)+ "%\n")
        pickle.dump((updates,cost,layer0,test_model,predictions,conditional_dist),open("results/"+str(batch_size)+representation+str(history)+".model","w"))
        
            #if patience <= iter:
            #    done_looping = True
            #    break

    fw.close()
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




def getBatch(names,batch_i,cumLengths,batch_size,representation='raw',batchType='random',history=0):
    if batchType=='random':
        return getRandBatch(names,batch_size,representation,history)
    else:
        return getFastBatch(names,batch_i,cumLengths,batch_size,representation,history)

def getRandBatch(names,batch_size,representation='raw',history=0):
    fs = random.sample(names,batch_size)
    ls = map(lambda x: random.randint(1,len(readGame.readGAME(x))-1),fs)

    bls = map(lambda (f,l): readGame.processGAMEatMove(f,moveN=l,gs=gs), zip(fs,ls))
    return utils.prepare_data(map(lambda game: reduce(lambda a,b: a+b,game),zip(*bls)),representation=representation,board_size=gs)

def getFastBatch(names, batch_i, cumLengths, batch_size,representation='raw',history=0):
     sg = batch_size*batch_i
     eg = batch_size*(batch_i + 1)
     si=0
     ei=1
     for (i,n) in enumerate(cumLengths):
         if n>sg:
            si = i
            break
     for (i,n) in enumerate(cumLengths):
         if n>eg:
            ei = i
            break
       
     fi=0
     ci=0     
     if si>0:
        fi = si
        ci = sg -cumLengths[si-1]
     x,y=readGame.processGAMEs(names[fi:ei+1],representation,history=history,gs=gs)
     return utils.prepare_data((x[ci:eg-cumLengths[ei]],y[ci:eg-cumLengths[ei]]),representation= representation,history=history,batch_size=batch_size,board_size=gs)

if __name__ == '__main__':
    default = "zero"
    if len(sys.argv)>1:
        default = sys.argv[1] 
    train(mynet=default)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)


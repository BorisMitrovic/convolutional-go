#!/bin/env python
from os import system

"""
My utility functions
"""

import scipy.io
import theano
import theano.tensor as T
import numpy

def convertLabels(y):
    return numpy.array(map(lambda x: numpy.where(x==1)[0][0],y))

def prepare_data(data_xy,batch_size=500,representation='raw',history=0,board_size=19):
    data_xs, data_y = data_xy
    data_xs = data_xs[:batch_size]
    data_y = data_y[:batch_size]
    
    data_xs = numpy.array(data_xs)
#    data_y = numpy.array(data_y)
    bs = board_size
    nd = 2 if representation=="raw" else 6
    bin_x = numpy.zeros((batch_size,bs*bs*nd*(history+1)))
    for (i,data_x) in enumerate(data_xs):
        data_x = data_x.reshape(data_x.shape[0],data_x.shape[1]*data_x.shape[2])
        n = data_x.shape[0]
    
#        data_x = numpy.array(data_x)
        for j in range(n):     
            if representation=="raw":
                bin_x[i,j*2*bs*bs:(j*2+1)*bs*bs] = data_x[j] ==1
                bin_x[i,(j*2+1)*bs*bs:(j*2+2)*bs*bs] = data_x[j] ==2
            elif representation =="liberty":
                bin_x[i,j*6*bs*bs:(j*6+1)*bs*bs] = data_x[j] < -2
                bin_x[i,(j*6+1)*bs*bs:bs*bs*(j*6+2)] = data_x[j] == -2
                bin_x[i,bs*bs*(j*6+2):bs*bs*(j*6+3)] = data_x[j] == -1
                bin_x[i,bs*bs*(j*6+3):bs*bs*(j*6+4)] = data_x[j] == 1
                bin_x[i,bs*bs*(j*6+4):bs*bs*(j*6+5)] = data_x[j] == 2
                bin_x[i,bs*bs*(j*6+5):bs*bs*(j*6+6)] = data_x[j] > 2
    
    return bin_x,data_y

def prepare_single(board,batch_size=10,representation='raw',history=0,board_size=5,shared=True):
    # NOTE: history fixed to 0
    bs = board_size
    nd = 2 if representation=="raw" else 6
    bin_x = numpy.zeros((batch_size,bs*bs*nd*(history+1)))
    
    data_x = board.reshape(board.shape[0]*board.shape[1])
    i=0
    j=0
    if representation=="raw":
        bin_x[i,j*2*bs*bs:(j*2+1)*bs*bs] = data_x ==1
        bin_x[i,(j*2+1)*bs*bs:(j*2+2)*bs*bs] = data_x ==2
    elif representation =="liberty":
        bin_x[i,j*6*bs*bs:(j*6+1)*bs*bs] = data_x < -2
        bin_x[i,(j*6+1)*bs*bs:bs*bs*(j*6+2)] = data_x == -2
        bin_x[i,bs*bs*(j*6+2):bs*bs*(j*6+3)] = data_x == -1
        bin_x[i,bs*bs*(j*6+3):bs*bs*(j*6+4)] = data_x == 1
        bin_x[i,bs*bs*(j*6+4):bs*bs*(j*6+5)] = data_x == 2
        bin_x[i,bs*bs*(j*6+5):bs*bs*(j*6+6)] = data_x > 2
    
    shared_x = theano.shared(numpy.asarray(bin_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    if shared:
        return shared_x
    else:
        return bin_x



def prepareFuego(txt,representation='raw'):   
    try: 
        a = numpy.array(map(int,txt[1:-3].split(" "))) # assuming txt = "[2 2 1 ... 0 2 ]"
        a.reshape(19,19)
        a[a==0] = 5
        a[a==1] = 6
        a[a==2] = 0
        a[a==5] = 1
        a[a==6] = 2
        
        data_x=a
        
        nd = 2 if representation=="raw" else 6
        history = 0
        bin_x = numpy.zeros((361*nd*(history+1)))
        j=0
        
        if representation=="raw":
            bin_x[j*2*361:(j*2+1)*361] = data_x ==1
            bin_x[(j*2+1)*361:(j*2+2)*361] = data_x ==2
        elif representation =="liberty":
            bin_x[j*6*361:(j*6+1)*361] = data_x < -2
            bin_x[(j*6+1)*361:361*(j*6+2)] = data_x == -2
            bin_x[361*(j*6+2):361*(j*6+3)] = data_x == -1
            bin_x[361*(j*6+3):361*(j*6+4)] = data_x == 1
            bin_x[361*(j*6+4):361*(j*6+5)] = data_x == 2
            bin_x[361*(j*6+5):361*(j*6+6)] = data_x > 2
        
        return bin_x.reshape(1,(361*nd*(history+1)))
    except ValueError:
        print "ERROR INPUT= '"+ txt + "'"
        raise

def move2fuego(move):
    return move+21 + move/19

def shared_dataset(data_xy, borrow=True,batch_size=500,board_size=19):
    """ Function that loads the dataset into shared variables

    The reason the dataset is stored in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = prepare_data(data_xy,batch_size=batch_size,board_size=board_size)
    
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, shared_y #T.cast(shared_y, 'int32')

def load_my_data(data="go.mat"):
    data = scipy.io.loadmat(data)
    aX = data['aX']
    ay = data['ay']
    
    aX = aX[:,:722]
    ay = ay[:,:722]
    
    vX = aX[:20000]
    vy = ay[:20000]
    
    tX = aX[20000:40000]
    ty = ay[20000:40000]
    
    X = aX[40000:]
    y = ay[40000:]
    
    train_set_x, train_set_y = shared_dataset(( X, y))
    test_set_x , test_set_y  = shared_dataset((tX,ty))
    valid_set_x, valid_set_y = shared_dataset((vX,vy))
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]
    return rval

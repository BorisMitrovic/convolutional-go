#!/usr/bin/env python
import pickle,utils,readGame,numpy,theano,my_net,time
from signal import signal, SIGPIPE, SIG_DFL
#signal(SIGPIPE,SIG_DFL) 

rets = []

def computeMRR(probs,y):
    return 1.0/ sum(map(lambda p: 1 if p>=probs[y] else 0,probs))
   
def computeAcc(pred,y):
    return 1 if numpy.argmax(pred)==y else 0
  
    
model=pickle.load(open("../convolutional/results/100raw4.model","r"))
(updates,cost,layer0,layer1,layer3,test_model,predictions,conditional_dist) = model


(trainCumLengths,validCumLengths,testCumLengths,filenames) = pickle.load(open("../convolutional/results/lengths.cache",'r'))
fn = filenames[:1000]
fncl = trainCumLengths[:1000]
batch_size = 1

valid_batch_x, valid_batch_y = utils.shared_dataset(readGame.processGAMEs(filenames[:6],'raw'))
test_batch_x, test_batch_y = utils.shared_dataset(readGame.processGAMEs(filenames[:6],'raw'))

    game # get game
    # set batch size to 1
    vx = utils.shared_dataset(game,representation='raw')
    #vx,vy = my_net.getBatch(fn, i, fncl, batch_size,'raw',batchType='fast',history=0)
    valid_batch_x.set_value(vx)
    
    #conds=numpy.array(conditional_dist())
    #move= numpy.argmax(conds)
    move = predictions()[0]
    move= utils.move2fuego(move)
    rets.append(move)
    fw = open("py2c","w")
    fw.write(str(move))
    try:
        fw.close()
    except IOError:
        fw.close()
    if c%100 ==0:
        print "rets = " + str(rets)
        rets = []
        c=1





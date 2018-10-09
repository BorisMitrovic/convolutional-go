#!/usr/bin/env python
import pickle,utils,readGame,numpy,theano,my_net,time
from signal import signal, SIGPIPE, SIG_DFL
#signal(SIGPIPE,SIG_DFL) 

rets = []

def computeMRR(probs,y):
    return 1.0/ sum(map(lambda p: 1 if p>=probs[y] else 0,probs))
   
def computeAcc(pred,y):
    return 1 if numpy.argmax(pred)==y else 0
  
    
model=pickle.load(open("results/1raw0.model","r"))
(updates,cost,layer0,layer1,layer3,test_model,predictions,conditional_dist) = model


(trainCumLengths,validCumLengths,testCumLengths,filenames) = pickle.load(open("results/lengths.cache",'r'))
fn = filenames[:2000]
fncl = trainCumLengths[:2000]
batch_size = 1

valid_batch_x, valid_batch_y = utils.shared_dataset(readGame.processSGFs(filenames[:6],'raw'))
test_batch_x, test_batch_y = utils.shared_dataset(readGame.processSGFs(filenames[:6],'raw'))

c=0

while(True):
    c+=1;
    fr = open("c2py","r")
    txt = fr.read()
    fr.close()
    if txt=='':
        print "EMPTY INPUT"
        raise IOError
    #print "INPUT= '"+ txt + "'"
    #print "txt = " + txt
    vx = utils.prepareFuego(txt,representation='raw')
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

#acc = 0

#for i in range(nBatches):
#    vx,vy = my_net.getBatch(fn, i, fncl, 500,'raw',batchType='fast',history=4)
#    valid_batch_x.set_value(vx)
#    conds=numpy.array(conditional_dist())   # conddist seem to be no better than random
#    acc +=sum(map(lambda (ps,y): computeAcc(ps,y), zip(conds,vy)))/float(len(vy))
#print "Acc = " + str(acc/nBatches)





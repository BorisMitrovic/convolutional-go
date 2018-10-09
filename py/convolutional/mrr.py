#!/bin/env python
import pickle,utils,readGame,numpy,theano

def computeMRR(probs,y):
    return 1.0/ sum(map(lambda p: 1 if p>=probs[y] else 0,probs))
   
def computeAcc(pred,y):
    return 1 if numpy.argmax(pred)==y else 0
  
    
model=pickle.load(open("results/raw4.model","r"))
(updates,cost,layer0,layer1,layer3,test_model,predictions,conditional_dist) = model


(trainCumLengths,validCumLengths,testCumLengths,filenames) = pickle.load(open("results/raw.cache",'r'))
fn = filenames[:1000]
fncl = trainCumLengths[:1000]

valid_batch_x, valid_batch_y = utils.shared_dataset(readGame.processSGFs(filenames[:6],'raw'))
test_batch_x, test_batch_y = utils.shared_dataset(readGame.processSGFs(filenames[:6],'raw'))

nBatches = 408
mrr = 0

for i in range(nBatches):
    vx,vy = m.getBatch(fn, i, fncl, 500,'raw',batchType='fast',history=4)
    valid_batch_x.set_value(vx)
    conds=numpy.array(conditional_dist())
    mrr +=sum(map(lambda (ps,y): computeMRR(ps,y), zip(conds,vy)))/float(len(vy))
print "MRR = " + str(1/(mrr/nBatches))

acc = 0

for i in range(nBatches):
    vx,vy = m.getBatch(fn, i, fncl, 500,'raw',batchType='fast',history=4)
    valid_batch_x.set_value(vx)
    conds=numpy.array(conditional_dist())   # conddist seem to be no better than random
    acc +=sum(map(lambda (ps,y): computeAcc(ps,y), zip(conds,vy)))/float(len(vy))
print "Acc = " + str(acc/nBatches)

acc = 0

for i in range(nBatches):
    vx,vy = m.getBatch(fn, i, fncl, 500,'raw',batchType='fast',history=4)
    valid_batch_x.set_value(vx)
    valid_batch_y.set_value(vy)
    preds=numpy.array(predictions())        # predictions seem to be random
    acc +=sum(map(lambda (ps,y): 1 if ps==y else 0, zip(preds,vy)))/float(len(vy))
print "Acc from predictions = " + str(acc/nBatches)


acc = 0

for i in range(nBatches):
    vx,vy = m.getBatch(fn, i, fncl, 500,'raw',batchType='fast',history=4)
    print vy[0]
    test_batch_x.set_value(vx)
    test_batch_y.set_value(vy)
    print test_model()
    acc +=test_model()          # doesn't update test_batch_x and y
print "Acc from test_model() = " + str(acc/nBatches)











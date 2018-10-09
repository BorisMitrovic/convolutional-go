#!/usr/bin/env python

import re, time, pickle
import readGame
import subprocess as sp
from UCT import *



def evalMC(sgffile):
    moves = readGame.readGAME(sgffile)
    
    c=0
    state = CapGoState(5)
    for move in moves:
        n = UCT(rootstate = state, itermax = 1000, verbose = False)
        if n.move == move:
            c+=1
        state.DoMove(move)
        
    result = (c,len(moves))
    print "Accuracy is: " + str(result[0] / float(result[1]))
    return result    

def evalTest():
    datasetMY="games"
    fn1 = readGame.getFilenames(datasetMY,1,0,1)[0]
    files = fn1[:100]
    print "Evaluating " +str(len(files)) + " files, estimating process time is " + str(len(files)/2) + " hours"
    start = time.time()
    results = map(evalMC,files)
    result = map(lambda s: reduce(lambda a,b: a+b,s),zip(*results))
    tt = time.time()-start
    print "Accuracy is: " + str(result[0] / float(result[1])*100) + "%, time taken is: " + str(tt)


if __name__ == "__main__":
#    sampleSGF = "../../pro-GoGod/2014/2014-01-03a.sgf"
#    start = time.time()
#    result = evalFuego(sampleSGF)
#    tt = time.time()-start
#    print "Accuracy is: " + str(result[0] / float(result[1])*100) + "%, time taken is: " + str(tt)
    evalTest()

#!/usr/bin/env python

import re, time, pickle
import readGame
import subprocess as sp

coords = "ABCDEFGHJKLMNOPQRST" 

def moveCoor2Fuego(move):
    return coords[move[0]]+str(move[1]+1)


def evalFuego(sgffile):
    moves = readGame.readSGF(sgffile)
    moves = map(moveCoor2Fuego,moves)
#    moves = moves[:5]
    plays = map(lambda (i,move): "play "+"bw"[i%2]+" "+move,enumerate(moves))
    commands = "\n".join(map(lambda (i,play): "genmoveM "+ "bw"[i%2]+"\n"+play,enumerate(plays)))
    
    fw = open("callFuego.sh","w")
    begin= """#!/bin/bash
fuego << EOF
"""
    core = commands
    end = """
EOF
""" 
    txt = begin + core + end
    
    fw.write(txt)
    fw.close()
    output =  sp.check_output(['./callFuego.sh'])
    predictions = re.findall("= (\w\d+)",output)
    result = (sum(map(lambda(p,t): 1 if p==t else 0, zip(predictions,moves))),len(predictions))
    print "Accuracy is: " + str(result[0] / float(result[1]))
    predfile = re.findall("\/[^\/]*$",sgffile)[0][1:-4]+".pred"
    fw = open("predictions/"+predfile,"w")
    fw.write(" ".join(predictions))
    fw.close()
    return result    

def evalTest():
    (trainCumLengths,validCumLengths,testCumLengths,filenames) = pickle.load(open("results/raw.cache",'r'))
    files = filenames[int(0.999*len(filenames)):]
    print "Evaluating " +str(len(files)) + " files, estimating process time is " + str(len(files)/2) + " hours"
    start = time.time()
    results = map(evalFuego,files)
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

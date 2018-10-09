#!/usr/bin/env python
from os import system

import re
import numpy
import os
import random

nProcessed = 0
board_size = 5  #TODO: important - change to bs if want to use my_net
bs = board_size

def convertMove(move):
    move = tuple(map(lambda x: ord(x)-ord('a'), move))
    return move

def readSGF(fname):
    
    fr=open(fname,'r')
    txt = fr.read()[:-1]
    #last = len(txt)-txt[::-1].index('\n')
    try:
        last = txt.index(';B[')
    except ValueError:
        return []
    
    game = txt[last:-1]
    
    pattern = re.compile('\[(\w\w)\]')
    moves = map(convertMove,pattern.findall(game))
    
    # ignoring incorrectly formatted moves
    #t = filter(lambda (i,(x,y)): x>=bs or y>=bs or x<0 or y<0, enumerate(moves))
    #if t !=[]:      # invalid move - cut the game to only include prev moves
    #    moves = moves[:t[0][0]]
    #    print "Game " + fname + " cut at move " + str(t[0][0])
    
    return moves
    
def readGAME(fname):
    fr=open(fname,'r')
    txt = fr.read()
    moves = eval(txt)
    return moves    

def createPosition(moves):
    board = numpy.zeros((bs, bs))
    n = 0
    for move in moves:
        playMove(move,board,n)
        n+=1
    return board

def convertLiberty(board):
    groups = findGroups(board)
    libs = map(lambda g: countLibs(board,g),groups)
    gls = zip(groups,libs)
    for (group,nlib) in gls:
        (colour,stones) = group
        for stone in stones:
            board[stone] = nlib*(colour*2-3) 

def countLibs(board,group):
    (colour,stones) = group
    libs = []
    for (x,y) in stones:
        for loc in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            if loc[0]>=0 and loc[1]>=0 and loc[0] < bs and loc[1] <bs and board[loc]==0:
                libs.append(loc)
    return len(list(set(libs)))
    
def findGroups(board):
    unassignedWhites = []
    unassignedBlacks = []
    (il,jl) = board.shape
    for i in range(il):
        for j in range(jl):
            if board[(i,j)]==1:
                unassignedBlacks.append((i,j))
            elif board[(i,j)]==2:
               unassignedWhites.append((i,j))
    whiteGroups = joinUnassigned(unassignedWhites,2)
    blackGroups = joinUnassigned(unassignedBlacks,1)
    return whiteGroups + blackGroups

def joinUnassigned(unassigned,colour):
    if unassigned == []:
        return []
    groups = [[unassigned[0]]]
    for stone in unassigned[1:]:
        for group in groups:
            if any(map(lambda s: nextTo(stone,s),group)):
                group.append(stone)
            else:
                groups.append([stone])
                break
    cond = True
    while (cond):
        cond = False
        for i in range(len(groups)-1):
            for j in range(len(groups)-i-1):
                if anyCommon(groups[i],groups[i+j+1]):
                    temp = groups.pop(i+j+1)
                    groups[i] = groups[i] + temp
                    cond = True
                    break
            if cond:
                break
    return map(lambda group: (colour,group), groups)
    
def anyCommon(g1,g2):
    for s1 in g1:
        for s2 in g2:
            if nextTo(s1,s2):
                return True
    return False
    
def nextTo(a,b):
    (a1,a2) = a
    (b1,b2) = b
    return abs(a1-b1) + abs(a2-b2) == 1    
    
def playMove(move,board,n,representation='raw'):
    justPlayed = n%2 + 1
    if representation=='raw':
        board[move] = justPlayed
        removeDead(board,move)
    elif representation=='liberty':
        updateLibs(board,move,justPlayed)

def convertRaw(board):
    board = (numpy.sign(board) + 1)/2+1

def updateLibs(board,move,justPlayed):
    sgn = justPlayed*2-3
    for neigh in neighbour(move):
        if board[neigh] == -sgn:
            # captured group
            group = getGroup(board,neigh)
            surrounding = []
            for stone in group:
                board[stone]=0
                surrounding += filter(lambda neigh: numpy.sign(board[neigh])==sgn, neighbour(stone))
                
            surrounding = list(set(surrounding))
            sgroups = map(lambda stone: getGroup(board,stone),surrounding)
            
            sgroups = set(tuple(x) for x in sgroups)
            sgroups = [ list(x) for x in sgroups ]
            
            for sgroup in sgroups:
                libs = countLibs(board,((sgn+1)/2+1,sgroup))
                for s in sgroup:
                    board[s]=libs
    
    for stone in growGroup(board,filter(lambda neigh: numpy.sign(board[neigh])==-sgn, neighbour(move))):
        board[stone]+=sgn
    
    board[move]=sgn
    group = getGroup(board,move)
    libs = countLibs(board,((sgn+1)/2+1,group))
    for stone in group:
        board[stone]=libs*sgn

def getGroup(board,move):
    group = [move]
    return growGroup(board,group)

def growGroup(board,group): #TODO: fixme
    newgroup = group
    for stone in group:
        newgroup = newgroup+filter(lambda neigh: numpy.sign(board[neigh])==numpy.sign(board[stone]), neighbour(stone))
    newgroup = list(set(newgroup))
    if len(newgroup) == len(group):
        return group
    else:
        return growGroup(board,newgroup)
        
def neighbour(move):
    (x,y) = move
    moves = []
    for (i,j) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if i>=0 and j>=0 and i<bs and j<bs:
            moves.append((i,j))
    return moves

def doesnthavelibs(board,stone):
    (x,y) = stone
    res = False
    for (i,j) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if i>=0 and j>=0 and i<bs and j<bs:
            res = res or board[(i,j)]==0
    return not res

def removeDead(board,move):
    # assumes no suicides allowed
    colour = 3-board[move]
    for stone in neighbour(move):
        if board[stone]==colour: 
            hasLibs = colourLibertyless(board,stone,colour)
            if hasLibs:
                # not dead - revert
                board[board==4] = colour
            else:
                # remove dead
                board[board==4] = 0
        
def colourLibertyless(board,stone,colour):
    if board[stone]!=colour:
        return False
    if doesnthavelibs(board,stone):
        board[stone] = 4 # special colour for now
        for neigh in neighbour(stone):
            if board[neigh]==colour:
                hasLibs = colourLibertyless(board,neigh,colour)
                if hasLibs:
                    return True
        return False
    return True
                    
    
def outputBoards(boards,fname):
    fw = open(fname,'w')
    for board in boards:
        string = ' '.join(map(str,board.flatten()))
        fw.write(string+'\n')
    fw.close()

def outputLabels(labels,fname):
    fw = open(fname,'w')
    for label in labels:
        fw.write(str(label) + '\n')
    fw.close()
    
def processSGF(fname):
    global nProcessed
    if random.random()<0.01:
        nProcessed +=1
        print "processed: " + str(nProcessed) + "00 / 147962"
    moves = readSGF(fname)
    boards = []
    labels = []
    for i in range(len(moves)-1):
        boards.append(createPosition(moves[:i]))
        correct = moves[i]
        labels.append(correct[0]*bs + correct[1])
    return (boards,labels)

def swapColours(board,representation='raw'):
    if representation=='raw':
        return (3-board)%3
    else:
        return board*-1

def getCumGameLengths(filenames,ftype="sgf"):
    if ftype=="sgf":
        ls = map(lambda x: len(readSGF(x)),filenames)
    elif ftype=="game":
        ls = map(lambda x: len(readGAME(x)),filenames)
    c=0
    res = []
    for l in ls:
        c+=l
        res.append(c)
    return res

def processSGFatAllMoves(fname,representation='raw',history=0):
    moves = readSGF(fname)
    boards = []
    labels = []
    
    board = numpy.zeros((bs, bs))
    n = 0
    for move in moves:
        if n%2 == 0:
            boards.append(board.copy())
        else:
            boards.append(swapColours(board.copy(),representation))
        
        labels.append(move[0]*bs + move[1])
        playMove(move,board,n,representation=representation)
        n+=1

#    if representation=='liberty-slow':
#        for move in moves:
#            # TODO: use convertLiberty(board) to create a liberty representation
#            if n%2 == 0:
#                libBoard = board.copy()
#                convertLiberty(libBoard)    # TODO: replace this VERY SLOW method with an algorithm which updates the liberties only based on the previous move
#                boards.append(libBoard)
#            else:
#                libBoard = swapColours(board.copy(),'raw')
#                convertLiberty(libBoard)
#                boards.append(libBoard)
            
#            labels.append(move[0]*bs + move[1])
#            playMove(move,board,n)
#            n+=1
    
    histbds = [([numpy.zeros((bs,bs))]*(history-i) + boards[:i+1] if i<history else boards[i-history:i+1]) for (i,bd) in enumerate(boards)]
            
    return(histbds,labels)    

def processGAMEatAllMoves(fname,representation='raw',history=0):
    moves =readGAME(fname)
    boards = []
    labels = []
    
    board = numpy.zeros((5, 5))
    n = 0
    for move in moves:
        if n%2 == 0:
            boards.append(board.copy())
        else:
            boards.append(swapColours(board.copy(),representation))
        
        labels.append(move[0]*5 + move[1])
        playMove(move,board,n,representation=representation)
        n+=1
    
    histbds = [([numpy.zeros((5,5))]*(history-i) + boards[:i+1] if i<history else boards[i-history:i+1]) for (i,bd) in enumerate(boards)]
    return(histbds,labels) 


def processSGFs(fnames,representation='raw',history=0):
    bls = map(lambda fname: processSGFatAllMoves(fname,representation,history),fnames)
    (boards,labels) = map(lambda game: reduce(lambda a,b: a+b,game),zip(*bls))
    return (boards,labels)  

def processGAMEs(fnames,representation='raw',history=0):
    bls = map(lambda fname: processGAMEatAllMoves(fname,representation,history),fnames)
    (boards,labels) = map(lambda game: reduce(lambda a,b: a+b,game),zip(*bls))
    return (boards,labels)  
    
def processSGFatMove(fname,moveN=40):
    # do just move 40 for example!
    global nProcessed
    moves = readSGF(fname)
    boards = []
    labels = []
    i=moveN
    if len(moves) <= moveN:
        print "actual: " + str(len(moves)) + " req: " + str(moveN)
        return ([],[])
    board = createPosition(moves[:i])
    #convertLiberty(board)   # create liberty representation
    correct = moves[i]
    label = correct[0]*bs + correct[1]
    return ([board],[label])
    
def getFilenames(directory,ratio,split,nSplits):
    assert split <= nSplits
    assert ratio <=1 and ratio >=0
    fnames = []
    for path, subdirs, fname in os.walk(directory):
        fnames = fnames +map(lambda x: path+"/"+x,fname)
    nFiles = len(fnames)/nSplits
    nSplit = int(nFiles*ratio)
    start = nFiles * split
    end   = nFiles * (split+1)
    train = fnames[start:start+nSplit]
    test  = fnames[start+nSplit:end]
    return (train,test)
    
def convert(directory='../../go-data',ratio=0.8,nSplits=10,funct=processSGFatMove):
    
    for i in range(nSplits):
        print "\n\n   Iteration " + str(i)
        print "getting fnames..."
        (trainFiles,testFiles) = getFilenames(directory,ratio,i,nSplits)
        
        print "\n  Training set"
        print "processing SGF files..."
        bls = map(funct,trainFiles)
        print "reducing..."
        (boards,labels) = map(lambda game: reduce(lambda a,b: a+b,game),zip(*bls))
        print "outputing boards..."
        outputBoards(boards,'presence_train'+str(i)+'.txt')
        print "outputing labels..."
        outputLabels(labels,'presence_train'+str(i)+'.lbl')
        
        print "\n  Testing set"
        print "processing SGF files..."
        bls = map(funct,testFiles)
        print "reducing..."
        (boards,labels) = map(lambda game: reduce(lambda a,b: a+b,game),zip(*bls))
        print "outputing boards..."
        outputBoards(boards,'presence_test'+str(i)+'.txt')
        print "outputing labels..."
        outputLabels(labels,'presence_test'+str(i)+'.lbl')
    
if __name__ == "__main__":
    convert(directory='../../go-data',ratio=0.8,nSplits=1,funct=processSGFatMove)
    
    # train = ['go-data/2013-03-31-1.sgf','go-data/2013-03-31-2.sgf','go-data/2013-03-31-3.sgf','go-data/2013-03-31-4.sgf','go-data/2013-03-31-5.sgf','go-data/2013-03-31-6.sgf','go-data/2013-03-31-7.sgf','go-data/2013-03-31-8.sgf','go-data/2013-03-31-9.sgf']
    
    # test  = ['go-data/2013-03-30-1.sgf','go-data/2013-03-30-2.sgf','go-data/2013-03-30-3.sgf','go-data/2013-03-30-4.sgf','go-data/2013-03-30-5.sgf','go-data/2013-03-30-6.sgf','go-data/2013-03-30-7.sgf','go-data/2013-03-30-8.sgf','go-data/2013-03-30-9.sgf']
    
    
    
    

import re
import numpy
import os
import random

nProcessed = 0

def convertMove(move):
    move = tuple(map(lambda x: ord(x)-ord('a'), move))
    return move

def readSGF(fname):
    # last newline needs rethinking!
    # TODO: go-data/2004-02-26-22.sgf
    
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
    
    return moves
    
def createPosition(moves):
    board = numpy.zeros((19, 19))
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
            if loc[0]>=0 and loc[1]>=0 and loc[0] < 19 and loc[1] <19 and board[loc]==0:
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
    
def playMove(move,board,n):
    justPlayed = n%2 + 1
    board[move] = justPlayed
    removeDead(board,move)

def neighbour(move):
    (x,y) = move
    moves = []
    for (i,j) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if i>=0 and j>=0 and i<19 and j<19:
            moves.append((i,j))
    return moves

def doesnthavelibs(board,stone):
    (x,y) = stone
    res = False
    for (i,j) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if i>=0 and j>=0 and i<19 and j<19:
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
        labels.append(correct[0]*19 + correct[1])
    return (boards,labels)

def swapColours(board):
    return (3-board)%3

def processSGFatAllMoves(fname):
    moves = readSGF(fname)
    boards = []
    labels = []
    
    board = numpy.zeros((19, 19))
    n = 0
    for move in moves:
        # TODO: use convertLiberty(board) to create a liberty representation
        if n%2 == 0:
            boards.append(board.copy())
        else:
            boards.append(swapColours(board.copy()))
        
        labels.append(move[0]*19 + move[1])
        playMove(move,board,n)
        n+=1
    return(boards,labels)    

    
    
def processSGFatMove(fname,moveN=40):
    # do just move 40 for example!
    global nProcessed
    if random.random()<0.01:
        nProcessed +=1
        print "processed: " + str(nProcessed) + "00 / 147962"
    moves = readSGF(fname)
    boards = []
    labels = []
    i=moveN
    if len(moves) <= moveN:
        return ([],[])
    board = createPosition(moves[:i])
    convertLiberty(board)   # create liberty representation
    correct = moves[i]
    label = correct[0]*19 + correct[1]
    return ([board],[label])
    
def getFilenames(directory,ratio,split,nSplits):
    assert split <= nSplits
    assert ratio <=1 and ratio >=0
    fnames = map(lambda fname: directory+"/"+fname,os.listdir(directory))
    nFiles = len(fnames)/nSplits
    nSplit = int(nFiles*ratio)
    start = nFiles * split
    end   = nFiles * (split+1)
    train = fnames[start:start+nSplit]
    test  = fnames[start+nSplit:end]
    return (train,test)
    
def convert(directory='go-data',ratio=0.8,nSplits=10,funct=processSGFatMove):
    
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
    convert(directory='go-data',ratio=0.8,nSplits=1,funct=processSGFatMove)
    
    # train = ['go-data/2013-03-31-1.sgf','go-data/2013-03-31-2.sgf','go-data/2013-03-31-3.sgf','go-data/2013-03-31-4.sgf','go-data/2013-03-31-5.sgf','go-data/2013-03-31-6.sgf','go-data/2013-03-31-7.sgf','go-data/2013-03-31-8.sgf','go-data/2013-03-31-9.sgf']
    
    # test  = ['go-data/2013-03-30-1.sgf','go-data/2013-03-30-2.sgf','go-data/2013-03-30-3.sgf','go-data/2013-03-30-4.sgf','go-data/2013-03-30-5.sgf','go-data/2013-03-30-6.sgf','go-data/2013-03-30-7.sgf','go-data/2013-03-30-8.sgf','go-data/2013-03-30-9.sgf']
    
    
    
    

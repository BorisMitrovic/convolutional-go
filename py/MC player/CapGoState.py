
from ctypes         import *
from math import *
import random

class CapGoState:
    """ A state of the game of Capture Go, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 
        2 = player 2 (O).
        In Capture Go players alternately place pieces on a square board - 
        each piece played has to be placed on an unocuppied intersection.
        If any group of stones has zero liberties, it is captured. If any of 
        player's groups is captured, that player looses the game. 
    """
    def __init__(self,sz = 9):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.precedingMoves = []
        self.prevMove = None
        self.prev2Moves = (None,None)
        self.board = [] # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        self.result=0
        assert sz == int(sz) and sz > 1 # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = CapGoState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        st.result = self.result
        st.precedingMoves = [move for move in self.precedingMoves]
        return st
        
    def GetPrevMove(self):
        """ Get the move which led to this state. 
        """
        return self.prevMove
    def GetPrev2Moves(self):
        """ Get the move which led to this state. 
        """
        return self.prev2Moves

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        #print self.precedingMoves
        #print move
        (x,y)=(move[0],move[1])
        if not (x == int(x) and y == int(y) and self.IsOnBoard(x,y) and self.board[x][y] == 0): 
            print self.precedingMoves + [move]
            print self
            
        assert x == int(x) and y == int(y) and self.IsOnBoard(x,y) and self.board[x][y] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        (f,s) = self.prev2Moves
        self.prev2Moves = (move,f)
        self.prevMove = move
        self.precedingMoves.append(move)
    
    
    def GetMoves(self):
        
        moves = self.GetPrecedingMoves()
        size = self.size
        
        """ TODO: should be done only once? """
        wd = u'/home/boris/Desktop/PhD/py/MC player'
        clib = CDLL(wd+'/capture.so')
        moves = map(lambda (x,y): x*size+y,moves)
        
        #fw=open("moves.txt","w")
        #fw.write(str(moves))
        #fw.close()
        szsz=size*size
        res = (c_int*szsz)()
        n = len(moves)
        arr = (c_int*n)()
        for i in range(n):
            arr[i] = moves[i]
        #print "Python gives input"
        a = clib.getLegalMoves(arr,n,res)
        #print "Python receives output\n"
        
        if a > 0:   # winning move
            a-=1;
            self.result= 3-self.playerJustMoved
            return [(a/size,a%size)]
            
        if a==-1:   # lost position
            self.result= self.playerJustMoved
            return []
            
        #translate back        
        r=[(move/size,move%size) for move in range(size*size) if res[move]==1]
        if r==[]:
            self.result= self.playerJustMoved
        return r
    
    
    """
    def GetMoves(self):   
        if self.GetCaptured():
            return []
        else:
            return [(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0]
    """
    
    def GetPrecedingMoves(self):
        return self.precedingMoves
        
    def GetCaptured(self):
        captured = []
        for x in range(self.size):
            for y in range(self.size):
                inter = self.board[x][y]
                if inter !=0:
                    neighs = self.GetNeighbours(x,y)
                    more = True
                    while(more):
                        more = False
                        vals = self.Vals(neighs)
                        if 0 in vals:
                            more = True
                            break
                        for i in range(len(vals)):
                            if vals[i] == inter:
                                potNeighs = self.GetNeighbours(neighs[i][0],neighs[i][1])
                                newNeighs = list(set(neighs+potNeighs))
                                if set(newNeighs) != set(neighs): 
                                    neighs = newNeighs
                                    more = True
                                    break
                    if not more:
                        captured = captured + [inter]       # only colour, not coordinates
                            
        return captured

    def Vals(self,neighs):
        return map(lambda (x,y): self.board[x][y],neighs)

    def GetNeighbours(self,x,y):
        potential = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        return filter(lambda (x,y): self.IsOnBoard(x,y),potential)

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        assert self.result !=0
        return 1-abs(self.result-playerjm)
        
    def __repr__(self):
        s= ""
        for y in range(self.size-1,-1,-1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s
      


from math import *
import random

class PhutState:
    """ A state of the game of Phutball (Philosopher's football), i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = black (X), 
        2 = white (O).
        In Phutball players alternately place black pieces on a square board 
        (each piece played has to be placed on an unocuppied intersection), 
        or they jump with the single white piece.
        If white stone jumps off the player's edge, he wins. 
        Y-axis is the winning axis
    """
    def __init__(self,sz = 9):
        # first player needs to reach 0 or -1, second player needs to reach sz or sz-1
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.prevMove = None
        self.prev2Moves = (None,None)
        self.board = [] # 0 = empty, 1 = black, 2 = single white stone
        self.size = sz
        assert sz == int(sz) and sz > 1 # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)
        self.board[sz/2][sz/2] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = PhutState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def GetPrevMove(self):
        """ Get the move which led to this state. 
        """
        return self.prevMove
        
    def GetPrev2Moves(self):
        """ Get the move which led to this state. 
        """
        return self.prev2Moves
        
    def DoMove(self, (colour,move)):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        if colour == 1:     # black
            (x,y)=(move[0],move[1])
            assert x == int(x) and y == int(y) and self.IsOnBoard(x,y) and self.board[x][y] == 0
            self.board[x][y] = 1
        else:               # white
            start = self.GetWhite()
            end = self.Cap(move[-1])
            self.board[start[0]][start[1]] = 0
            for jump in move:
                self.DoJump(start,jump)                
                start = jump
            self.board[end[0]][end[1]] = 2
        
        self.playerJustMoved = 3 - self.playerJustMoved
        (f,s) = self.prev2Moves
        self.prev2Moves = (move,f)
    
    def Cap(self,(x,y)):
        high = self.size-1
        
        ny = 0 if y<0 else y
        ny = high if y>high else ny
        return (x,ny)
    
    def GetInbetween(self,start,end):
        (sx,sy) = start
        (ex,ey) = end
        minx = min(sx,ex)
        miny = min(sy,ey)
        maxx = max(sx,ex)
        maxy = max(sy,ey)
        difx = maxx-minx
        dify = maxy-miny
        nBlacks = max(difx,dify)
        assert difx == nBlacks or difx == 0
        assert dify == nBlacks or dify == 0
        bxs =[sx]*nBlacks if difx==0 else  range(minx+1,maxx) # BUG: both diagonals! maybe fixed
        bys =[sy]*nBlacks if dify==0 else  range(miny+1,maxy) # BUG: both diagonals!
        if copysign(1,sx-ex) != copysign(1,sy-ey):
            bys.reverse()
        return zip(bxs,bys)
        
                
    def DoJump(self,start,end):
        #remove all black stones in the way
        blacks = self.GetInbetween(start,end)
        for (bx,by) in blacks:
            assert self.board[bx][by] == 1
            self.board[bx][by] = 0
                
    def GetWhite(self):
        for i in range(self.size):
            for j in range(self.size):
                if 2 == self.board[i][j]:
                    return (i,j)
        assert False
        
    def OnWinningPos(self):
        (x,y) = self.GetWhite()
        return y<1 or y>self.size-2
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        if self.OnWinningPos():
            return []
        else:
            blacks = [(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0]
            whites = self.GetWhiteMoves()
            mb = map(lambda m: (1,m),blacks)
            mw = map(lambda m: (2,m),whites)
            return mw+mb

    def GetWhiteMoves(self):
        (sx,sy) = self.GetWhite()
        moves = []
        for (dx,dy) in [(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1),(-1,0),(-1,+1)]:
            if self.IsOnBoard(sx+dx,sy+dy) and self.board[sx+dx][sy+dy] == 1:
                nStones=1
                while self.IsOnBoard(sx+nStones*dx,sy+nStones*dy) and self.board[sx+nStones*dx][sy+nStones*dy] == 1:
                    nStones += 1
                if not self.IsOnBoard(sx+nStones*dx,sy+nStones*dy):
                    capped = self.Cap((sx+nStones*dx,sy+nStones*dy))
                    if self.IsOnBoard(capped[0],capped[1]):
                        moves += [[(sx+(nStones-1)*dx,sy+(nStones-1)*dy)]]
                    #else: illegal
                elif self.board[sx+nStones*dx][sy+nStones*dy] == 0:
                    nstate = self.Clone()
                    move =  [(sx+nStones*dx,sy+nStones*dy)]
                    nstate.DoMove((2,move))
                    moves += [move]
                    nmoves = nstate.GetWhiteMoves()
                    moves += map(lambda nmove: move + nmove,nmoves)
        return moves
        


    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        assert self.OnWinningPos()
        (x,y) = self.GetWhite()
        winner = 1 if y<1 else 2
        return 1-abs(winner-playerjm)

    def __repr__(self):
        s= ""
        for y in range(self.size-1,-1,-1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s
      

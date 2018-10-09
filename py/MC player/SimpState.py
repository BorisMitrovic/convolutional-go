
from math import *
import random

class SimpState:
    """ A state of the game Simp. In Simp, players alternately take 1 or 2 chips from 
        their pile with the 
        winner being the player to take the last chip from his pile. 
        In Simp taking 2 chips on every action is optimal (and one action of taking 1
        chip if an odd number of chips in player's pile).
        if chTop > chBot -1 more or less win for starting player TODO.
    """
    def __init__(self, (chTop,chBot)):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.prevMove = None
        self.prev2Moves = (None,None)
        self.chips = (chTop,chBot) # tuple
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = SimpState(self.chips)
        st.playerJustMoved = self.playerJustMoved
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
        assert move >= 1 and move <= 2 and move == int(move) and move <= self.chips[3-self.playerJustMoved-1]
        self.playerJustMoved = 3 - self.playerJustMoved
        self.chips = self.chips[0:self.playerJustMoved-1] +(self.chips[self.playerJustMoved-1]-move,) + self.chips[self.playerJustMoved:2]
        (f,s) = self.prev2Moves
        self.prev2Moves = (move,f)
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        if self.chips[self.playerJustMoved-1] == 0:
            return []
        else:
            return range(1,min([2, self.chips[3-self.playerJustMoved-1]])+1) 
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        # print str(self.chips) + " player: " + str(self.playerJustMoved)
        assert self.chips[self.playerJustMoved-1] == 0
        if self.playerJustMoved == playerjm:
            return 1.0 # playerjm took the last chip and has won
        else:
            return 0.0 # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Top:" + str(self.chips[0]) + " Bot:" + str(self.chips[1]) + " JustPlayed:" + str(self.playerJustMoved)
        return s


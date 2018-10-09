from math 	    	import *
from NimState 		import NimState
from SimpState 		import SimpState
from OXOState 		import OXOState
from OthelloState	import OthelloState
from CapGoState	    import CapGoState
from PhutState      import PhutState
import random, time, numpy, utils, pickle


def SHOT(rootstate,budget,budgetUsed,playouts,wins):
    node = Node(state = rootstate)
    node.budget = budget
    node.budgetUsed=budgetUsed
    if node.untriedMoves == [] and node.childNodes == []:   # terminal state
        backpropagate(node,state)
        return
    if budget ==1:     # last playout
        playout(state)
        node.budgetUsed +=1
        backpropagate(node,state)
        return
    if len(node.untriedMoves)+len(node.childNodes)==1:      # only move
        move = (node.untriedMoves+node.childNodes)[0]
        u=0
        p=0
        w=0
        state = rootstate.clone()
        state.DoMove(move)
        SHOT(state,budget,u,p,w)
        #TODO: update correctly - think
        return move
    if node.budget-node.budgetUsed <= len(node.untriedMoves)+len(node.childNodes):
        for move in node.untriedMoves:
            state=rootstate.clone()
            state.DoMove(move)
            playout(state)
            node.budgetUsed +=1
            backpropagate(node,state)
            if node.budgetUsed ==node.budget:
                return
    
    b=0
    S = node.untriedMoves + node.childNodes
    while len(S) > 1:
        b+= max(1,floor((2*node.budget-node.budgetUsed)/(len(S) * ceil(log2(len(node.untriedMoves)+len(node.childNodes)))))) 
        for move in S: 
            if 
    
    

def backpropagate(node,state):        
        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode
            
def playout(state):
    """ moves state by a random playout """ 
    while state.GetMoves() != []: # while state is non-terminal
        state.DoMove(random.choice(state.GetMoves()))

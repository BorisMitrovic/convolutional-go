# Change the game to be played in the UCTPlayGame() function at the bottom of the code.

from ctypes         import *
from math 	    	import *
from NimState 		import NimState
from SimpState 		import SimpState
from OXOState 		import OXOState
from OthelloState	import OthelloState
from CapGoState	    import CapGoState
from PhutState      import PhutState
import random, time, numpy, utils, pickle,sys

#valid_batch_x = utils.prepare_single(numpy.zeros((5,5)),board_size=5)
model=pickle.load(open("../convolutional/results/250raw0.model","r")) # trained on 12219 games UCT20k
(updates,cost,layer0,test_model,predictions,conditional_dist) = model
W = updates[0][0].get_value()   # weight vector
b = updates[1][0].get_value()   # biases

model=pickle.load(open("../convolutional/results/320raw0.model","r")) # trained on 20382 rosin10k games
(updates,cost,layer0,test_model,predictions,conditional_dist) = model
W1 = updates[0][0].get_value()   # weight vector
b1 = updates[1][0].get_value()   # biases argument


mixing = 0.01                   # mixing component for prior - obsolete

# IMPORTANT: every time change size here and in capture.c, and recompile capture.c
##############################################################################
sz = 5      ##################################################################
##############################################################################

class GameState:
    """ A state of the game. 
        The players are numbered 1 and 2.
    """
    def __init__(self):
            self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
            self.prevMove = None
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def GetPrevMove(self):
        """ Get the move which led to this state. 
        """
        return self.prevMove
        
    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        self.prevMove = move
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """

    def __repr__(self):
        pass

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.mu = 0.5
        self.sd = 0.2887    # math.sqrt(1/12.0)
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        self.result = state.result
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        expl = 1   
        self.childNodes.sort(key = lambda c: c.wins/float(c.visits) + expl*sqrt(2*log(self.visits)/float(c.visits)))
        return self.childNodes[-1]

    def PriorSelectChild(self,mixing=0.01):
        """ Combine prior and UCB1 formulas using a mixing factor
        """
        expl = 1   
        self.childNodes.sort(key = lambda c: self.prior[sz*c.move[0]+c.move[1]]**mixing * (c.wins/float(c.visits) + expl*sqrt(2*log(self.visits)/float(c.visits))))
        return self.childNodes[-1]

    def PriorSelectChildM(self):
        """ Combine prior and UCB1 formulas as in Rosin10 - also try sqrt(prior) normalised instead
        """
        expl = 1   
        self.childNodes.sort(key = lambda c: c.wins/float(c.visits) + sqrt(3/2.0*log(self.visits)/float(c.visits)) -2/self.prior[sz*c.move[0]+c.move[1]]*sqrt(log(self.visits)/self.visits) )
        return self.childNodes[-1]



    def BayesSelectChild(self):
        """ Use the Bayes formula presented by Tesauro10:
        """
        return max(self.childNodes,key = lambda c: c.wins/float(c.visits) + sqrt(2*log(self.visits)/float(c.visits)))
        #return max(self.childNodes,key = lambda c: c.mu + sqrt(2*log(self.visits)/float(c.visits)))
        #return max(self.childNodes,key = lambda c: c.mu + sqrt(2*log(self.visits))*c.sd)
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def GetRootNode(self):
        node = self
        while node != None:
            node1 = node
            node = node.parentNode
        return node1
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def UpdateBayes(self, result):
        """ Bayesian update of mean and standard deviation
        """
        if self.childNodes:
            (self.mu,self.sd) = reduce(maxBin,map(lambda x: (x.mu,x.sd),self.childNodes))
        else:
            self.mu = self.mu+self.sd*(result-self.mu)
            self.sd = self.sd*0.8

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s
   

def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited


def UCTc(rootstate, itermax, verbose = False):
    """ Use c code for fast playouts!
        Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate. 
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal 
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.result == 0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))
        
        if winner > 10:
            print "Incorrect node: "
            print state
            winner -= 10
            print ""

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:
        # decided position, as winning move exists
        n = rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
        return rootnode #n    
    
    return rootnode #toPlay[-1] # return the move that was most visited 
         
def getPrecedingMoves(node):
    movesRev = []
    while node != None:
        movesRev.append(node.move)
        node = node.parentNode
    movesRev = movesRev[:-1]
    return reversed(movesRev)
    
def BAYES(rootstate, itermax, verbose = False):
    """ Conduct a BAYES search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.BayesSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node.UpdateBayes(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited

def getPrior(board,toPlay):
    ##sx = utils.prepare_single(board,board_size=5,shared=False)
    ##valid_batch_x.set_value(sx) # FIXME: probably this doesn't update the value somehow
    ##prior = conditional_dist()[0]
    
    #prior = priorCompute(board)
    #print "BOARD: " + str(board)   
    #print "PRIOR: " + str(prior)
    return priorCompute(board,toPlay)

def softmax(w, t = 1.0):
    e = numpy.exp(numpy.array(w) / t)
    dist = e / numpy.sum(e)
    return dist

def binariseBoard(board,toPlay):
    board = numpy.array(board).reshape((sz*sz,))
    nb = numpy.zeros((sz*sz*2,))
    nb[:sz*sz] = board==toPlay
    nb[sz*sz:] = board==3-toPlay
    return nb

def priorCompute(board,toPlay):        
    r = numpy.dot(binariseBoard(board,toPlay),W)+b
    mask = numpy.array(board).reshape((sz*sz,))
    r[mask!=0]=numpy.NINF
    return softmax(r)

def priorComputeSqrt(board,toPlay):
    # sqrt probabilities        
    r = numpy.dot(binariseBoard(board,toPlay),W)+b
    mask = numpy.array(board).reshape((sz*sz,))
    r[mask!=0]=numpy.NINF
    r = map(lambda x: sqrt(x), softmax(r)) # sqrt probs
    mr = max(r)
    return map(lambda x: x/mr,r)   # normalise

def priorComputeSqrt2(board,toPlay):
    # sqrt probabilities        
    r = numpy.dot(binariseBoard(board,toPlay),W1)+b1
    mask = numpy.array(board).reshape((sz*sz,))
    r[mask!=0]=numpy.NINF
    r = map(lambda x: sqrt(x), softmax(r)) # sqrt probs
    mr = max(r)
    return map(lambda x: x/mr,r)   # normalise

def priorCompute2(board,toPlay):
    r = numpy.dot(binariseBoard(board,toPlay),W1)+b1
    mask = numpy.array(board).reshape((sz*sz,))
    r[mask!=0]=numpy.NINF
    return softmax(r)

def sampleFromPrior(priors):
    r = random.random()
    for (i,p) in enumerate(priors):
        r -= p
        if r <=0:
            return (i/sz,i%sz)
    raise ValueError

def rawNN(rootstate,itermax,verbose=False):
    rootnode = Node(state=rootstate)
    rootnode.prior = priorCompute(rootstate.board,3-rootstate.playerJustMoved)
    m = sampleFromPrior(rootnode.prior)
    state=rootstate.Clone()
    state.DoMove(m)
    node = Node(move = m, parent = rootnode, state = state)
    node.wins=1
    node.visits=1
    return node

def randomPlayer(rootstate,itermax,verbose=False):
    rootnode = Node(state=rootstate)
    m = random.choice(rootnode.untriedMoves) 
    node = Node(move = m, parent = rootnode, state = rootstate)
    node.wins=1
    node.visits=1
    return node

def rawNN2(rootstate,itermax,verbose=False):
    rootnode = Node(state=rootstate)
    rootnode.prior = priorCompute2(rootstate.board,3-rootstate.playerJustMoved)
    m = sampleFromPrior(rootnode.prior)
    state=rootstate.Clone()
    state.DoMove(m)
    node = Node(move = m, parent = rootnode, state = state)
    node.wins=1
    node.visits=1
    return node


def PRIORc(rootstate, itermax, verbose = False):
    """ Multiply prior information with the UCT value """


    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal
         
            node = node.PriorSelectChild(mixing)
            state.DoMove(node.move)


        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute(state.board,3-state.playerJustMoved)
            
        # Expand
        if node.result==0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree


        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:  
        # decided position, as winning move exists
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
    return toPlay[-1] # return the move that was most visited


def Rosin(rootstate, itermax, verbose = False):
    """ Use the Rosin formula to combine the prior information with the UCT value """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal
         
            node = node.PriorSelectChildM()
            state.DoMove(node.move)

        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute(state.board,3-state.playerJustMoved)
            
        # Expand
        if node.result==0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree


        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0) # state is terminal. Update node with result from POV of node.playerJustMoved
            # TODO: also update lookup based on win/loss
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:  
        # decided position, as winning move exists
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
    return toPlay[-1] # return the move that was most visited

def RosinSqrt(rootstate, itermax, verbose = False):
    """ Use the Rosin formula to combine the prior information with the UCT value """


    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal
         
            node = node.PriorSelectChildM()
            state.DoMove(node.move)


        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorComputeSqrt(state.board,3-state.playerJustMoved)
            
        # Expand
        if node.result==0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree


        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0) # state is terminal. Update node with result from POV of node.playerJustMoved
            # TODO: also update lookup based on win/loss
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:  
        # decided position, as winning move exists
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
    return toPlay[-1] # return the move that was most visited

def RosinSqrt2(rootstate, itermax, verbose = False):
    """ Use the Rosin formula to combine the prior information with the UCT value """


    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal
         
            node = node.PriorSelectChildM()
            state.DoMove(node.move)


        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorComputeSqrt2(state.board,3-state.playerJustMoved)
            
        # Expand
        if node.result==0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree


        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:  
        # decided position, as winning move exists
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
    return toPlay[-1] # return the move that was most visited



def PRIOR2c(rootstate, itermax, verbose = False):
    """ Multiply prior information with the UCT value, prior information lvl2 """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal
            node = node.PriorSelectChild(mixing)
            state.DoMove(node.move)

        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute2(state.board,3-state.playerJustMoved)
            
        # Expand
        if node.result==0 and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree


        # Rollout
        winner = node.result
        if winner ==0: # unknown, then random playout sample
            winner = cPlayout(state.precedingMoves,len(state.board))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(1 if winner==node.playerJustMoved else 0) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    toPlay = sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)
    if not toPlay:  
        # decided position, as winning move exists
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)
    return toPlay[-1] # return the move that was most visited
    
                
def UCTPlaySampleGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
    # state = OXOState() # uncomment to play OXO
    # state = NimState(15) # uncomment to play Nim with the given number of starting chips
    # state = SimpState((5,5)) # uncomment to play Simp with the given number of starting chips
    state = CapGoState(sz) # uncomment to play Capture Go with the given board size
    UCTPlayGame(state)

def UCTPlayNGames(state=CapGoState(sz),player1=UCTc,player2=UCTc,iter1=100,iter2=100,nGames=10):
    score = [0,0]
    startState = state.Clone()
    for i in range(nGames):
        state = startState.Clone()
        res = UCTPlayGame(state,player1,player2,iter1,iter2,verbose=False)  # assuming 1,0 nonadditive
        rb = int(res[0]+res[1]) % 2
        score[rb] +=1
        print str(score[0]) + " -vs- " + str(score[1])+ "\r",
        sys.stdout.flush()
    print "\npartial score"
    return score
                
def UCTPlayGame(state=CapGoState(sz), player1=UCTc,player2=UCTc,iter1=20000,iter2=20000,verbose=False):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes). An initial state is given
    """
    game = []
    while (state.GetMoves() != []):
        if (verbose): print str(state)
        if state.playerJustMoved == 2:
            n = player1(rootstate = state, itermax = iter1, verbose = verbose) # play with values for itermax and verbose = True
        else:
            n = player2(rootstate = state, itermax = iter2, verbose = verbose) 
        toPlay = sorted(n.childNodes, key = lambda c: c.visits+c.wins) 
        m = toPlay[-1].move 
        game.append((m,approxDist(n)))
        if (verbose): print "Best Move: " + str(m) + "  Confidence: "+ str(int(n.wins*100.0/n.visits)) + "\n"
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        if (verbose): print "Player " + str(state.playerJustMoved) + " wins!"
    elif state.GetResult(state.playerJustMoved) == 0.0:
        if (verbose): print "Player " + str(3 - state.playerJustMoved) + " wins!"
    elif (verbose): print "Nobody wins!"
    if (verbose): print "Game is: " + str(game)
    fw = open("gamesDist5/"+str(time.time())+".game","w")
    fw.write(str(game))
    fw.close()
    return (state.GetResult(state.playerJustMoved),state.playerJustMoved)

def approxDist(node):
    nw = sum(map(lambda c: c.wins, node.childNodes))
    nv = sum(map(lambda c: c.visits, node.childNodes))
    n = float(nv+nw)
    dist = [0]*sz*sz
    if n==0:
        dist[node.childNodes[0].move[0]*sz+node.childNodes[0].move[1]] = 1
        return dist    
    for c in node.childNodes:
        dist[c.move[0]*sz+c.move[1]]=(c.wins+c.visits)/n
    return dist

def maxBin((mu1,sd1),(mu2,sd2)):
    ro = 0.3 
    sdm = sd1**2-2*ro*sd1*sd2 + sd2**2
    alpha = (mu1-mu2)/sdm
    mu = mu2+sdm*alpha*FI(alpha) + fi(alpha)    # compute PDF and CDF
    sd = sqrt(abs((mu1**2+sd1**2)*FI(alpha)+(mu2**2+sd2**2)*FI(-alpha)+(mu1+mu2)*sdm*fi(alpha)-mu**2)) # abs is a hack, because sd^2 is negative for ((0.3,0.1),(0.4,0.3))
    #sd = sqrt(abs(sd2**2+(sd1**2-sd2**2)*FI(alpha) + sdm**2*alpha**2*FI(alpha)*(1-FI(alpha))+(1-2*FI(alpha))*alpha*fi(alpha)-fi(alpha)**2))  #TODO: this is meant to be a faster way to compute, but values different to the other way

    return (mu,sd)

def FI(x):  # cdf of N(0,1)
    return 0.5*erf(x/sqrt(2)) + 0.5

def fi(x):  # pdf of N(0,1)
    return (1/(sqrt(2*pi)))*exp(-x*x/2.0)
  
def confIntervalA(s,n,alpha=0.05):
    """ Agresti-Coull Interval """
    z=1-alpha/2.0
    nh = n + z**2
    ph = 1/nh * (s + z**2/2.0)
    CI = z*sqrt(1.0/nh*ph*(1-ph))
    print "p={:.2%} +- {:.2%}".format(ph,CI)
    return (ph,CI)

def confIntervalN(s,n,alpha=0.05):
    """ Normal approximation """
    z=1-alpha/2.0
    p = s/float(n)
    CI = z*sqrt(1.0/n*p*(1-p))
    print "p={:.2%} +- {:.2%}".format(p,CI)
    return (p,CI)

def testSpeed():
    st = time.time()
    n=10000
    for i in range(n):
        if i%1000==0:
            print i
        state = CapGoState(sz);
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))
    print (time.time()-st)

def cPlayout(moves,size):
    """ playout using C code """
    wd = u'/home/boris/Desktop/PhD/py/MC player'
    clib = CDLL(wd+'/capture.so')
    moves = map(lambda (x,y): x*size+y,moves)
    
    #fw=open("moves.txt","w")
    #fw.write(str(moves))
    #fw.close()
    
    n = len(moves)
    arr = (c_int*n)()
    for i in range(n):
        arr[i] = moves[i]
    #print "Python gives input"
    a = clib.playout(arr,n)
    #print "Python receives output\n"
    
    return a
    

def testC():
    wd = u'/home/boris/Desktop/PhD/py/MC player'
    clib = CDLL(wd+'/capture.so')
    n=3
    arr = (c_int*n)()
    res = (c_int*25)()
    arr[0]=1
    arr[1]=0
    arr[2]=5
    a=clib.getLegalMoves(arr,n,res)
    print "C returns " + str(a) + "!"

def testDiffParams(player1=rawNN,player2=UCTc,iter1=10000,iter2=10000,nGames=200):
    print "player1 = rosinSqrt2, player2 = rosinSqrt, nGames=" + str(nGames)
    for iter1 in [1,100,1000,10000]:
    #  for iter2 in [100,300,500,1000,3000,10000,20000]:
    #   if iter1 > iter2:
        iter2 = iter1 #int(round(iter1 *666.6667))
        print str(iter1) + " rosinSqrt2 vs rosinSqrt: " + str(iter2)
        res1 = UCTPlayNGames(state=CapGoState(sz), player1=player1,player2=player2,iter1=iter1,iter2=iter2,nGames=nGames/2)
        res2 = UCTPlayNGames(state=CapGoState(sz), player1=player2,player2=player1,iter1=iter2,iter2=iter1,nGames=nGames/2)

        confIntervalA((res1[0]+res2[1]), nGames)
        print "\n"

if __name__ == "__main__":
    #testDifferenceOfPrior()
    #testDiffParams()
    """ Play a single game to the end using UCT for both players. 
    """

    #"""
   
    i=0
    while(True):
        i+=1
        if i%100 == 0: print i
        UCTPlaySampleGame()
    #"""


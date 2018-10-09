
 
          
def AMAF(rootstate, itermax, verbose = False):
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

        root = node.GetRootNode()
        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            res = state.GetResult(node.playerJustMoved)
            node.Update(res) # state is terminal. Update node with result from POV of node.playerJustMoved
            # TODO: SOMETHING WRONG HERE! COUNTS ARE TOO SMALL!!!
            
            
            if node.move in root.untriedMoves:
                resstate = root.Clone()
                resstate.DoMove(node.move)
                root.AddChild(node.move,resstate)
            
            for cd in root.childNodes:
                if cd.move == node.move:
                    cd.wins   +=res
                    cd.visits +=1
                    break
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited



def LGRF1(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Uses LGRF-1 method described in Baier.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)
    lookup = {}

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
            # TODO: try lookup first
            try:
                move = lookup[state.GetPrevMove()]
                if move in state.GetMoves():
                    state.DoMove(move)
                    continue
            except KeyError:
                pass
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            res = state.GetResult(node.playerJustMoved)
            node.Update(res) # state is terminal. Update node with result from POV of node.playerJustMoved
            # TODO: also update lookup based on win/loss
            if (res==1):
                lookup[state.GetPrevMove()] = node.move
            elif (res==0):
                try:
                    lookup.pop(state.GetPrevMove()) 
                except KeyError:
                    pass
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited
    
    
    
def LGRF2(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Uses LGRF-2 method described in Baier.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)
    lookup1 = {}
    lookup2 = {}

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
            # TODO: try lookup first
            try:
                move = lookup2[state.GetPrev2Moves()]
                if move in state.GetMoves():
                    state.DoMove(move)
                    continue
            except KeyError:
                pass
            
            try:
                move = lookup1[state.GetPrevMove()]
                if move in state.GetMoves():
                    state.DoMove(move)
                    continue
            except KeyError:
                pass
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            res = state.GetResult(node.playerJustMoved)
            node.Update(res) # state is terminal. Update node with result from POV of node.playerJustMoved
            # TODO: also update lookup based on win/loss
            if (res==1):
                lookup1[state.GetPrevMove()] = node.move
                lookup2[state.GetPrev2Moves()] = node.move
            elif (res==0):
                try:
                    lookup1.pop(state.GetPrevMove()) 
                except KeyError:
                    pass
                
                try:
                    lookup2.pop(state.GetPrev2Moves())
                except KeyError:
                    pass    
                    
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited
    
def PRIOR(rootstate, itermax, verbose = False):
    """ Multiply prior information with the UCT value """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.PriorSelectChild(mixing)
            state.DoMove(node.move)

        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute(state.board,3-state.playerJustMoved)
            
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
            # TODO: also update lookup based on win/loss
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited
  
def PRIOR2(rootstate, itermax, verbose = False):
    """ Multiply prior information with the UCT value, prior information lvl2 """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.PriorSelectChild(mixing)
            state.DoMove(node.move)

        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute2(state.board,3-state.playerJustMoved)
            
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
            # TODO: also update lookup based on win/loss
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited
  

def PRIOR3(rootstate, itermax, verbose = False):
    """ Considers only top half of the likely moves """

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # First Filter out weaker half of the moves (using Prior Information)
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute(state.board,3-state.playerJustMoved)     #TODO
            allmoves = map(lambda x: divmod(x,5),range(5*5))
            ordered = sorted(zip(node.prior,allmoves),reverse=True)
            
            # try only top half of moves according to priors
            n = len(node.untriedMoves)
            n = max(1,n/2)
            selected = filter(lambda move: move in node.untriedMoves,map(lambda (x,y): y, ordered))[:n]
            node.untriedMoves = selected    # remove all other moves

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
            # TODO: also update lookup based on win/loss
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose=="tree"): print rootnode.TreeToString(0)
    elif (verbose=="children"): print rootnode.ChildrenToString()  
    
    
    return sorted(rootnode.childNodes, key = lambda c: c.visits+c.wins)[-1] # return the move that was most visited
 

def UCTcTest(rootstate, itermax, verbose = False):
    """ Use c code for fast playouts!
        Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate. 
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""


    global diffCountG
    global countG  
    global perplexity
    global countPerp   
    global priorAcc
    global countAcc  
    global MRR   
    global MNC 
    global crossEntropy
    global entropy

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and node.result==0: # node is fully expanded and non-terminal 
            no = node.UCTSelectChild()
            
            #TODO: remove me, just for testing the difference
            for i in range(12):
                mixing =[1000,100,10,1,.5,.2,.1,.05,.02,.01,.001,.0001][i]
                n = node.PriorSelectChild(mixing)
                mv = n.move
                diffCountG[i] += 1 if mv!=no.move else 0
            countG+=1  

            node = no
            state.DoMove(node.move)
            
        # Compute Prior once
        if node.untriedMoves != [] and node.childNodes == []:
            node.prior = priorCompute(state.board,3-state.playerJustMoved)
            countPerp +=1
            nonzeros = filter(lambda x: x!=0, node.prior)
            perp = -sum(map(lambda x: x*log(x)/log(2),nonzeros))
            perplexity = perplexity * ((countPerp-1)/float(countPerp)) + 1/float(countPerp) * perp
        #TODO: end remove me


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
        return rootnode.AddChild(rootnode.untriedMoves[0],rootstate)    
    
    #TODO: remove, just testing
    else:
        move = toPlay[-1].move
        priorAcc += 1 if rootnode.prior[move[0]*5+move[1]] == max(rootnode.prior) else 0
        countAcc += 1
        RR= 1.0/(len(filter(lambda x: x>rootnode.prior[move[0]*5+move[1]],rootnode.prior))+1)
        NC= 1.0/len(filter(lambda x: x>0,rootnode.prior))
        MRR = MRR * ((countAcc-1)/float(countAcc)) + 1/float(countAcc) * RR
        MNC = MNC * ((countAcc-1)/float(countAcc)) + 1/float(countAcc) * NC
        CE = -sum([node.visits/float(itermax) * log(rootnode.prior[node.move[0]*5+node.move[1]])/log(2) for node in rootnode.childNodes])
        EN = -sum([node.visits/float(itermax) * log(node.visits/float(itermax))/log(2) for node in rootnode.childNodes])
        crossEntropy = crossEntropy * ((countAcc-1)/float(countAcc)) + 1/float(countAcc) * CE
        entropy = entropy * ((countAcc-1)/float(countAcc)) + 1/float(countAcc) * EN
        
    #TODO: end remove
    
    return toPlay[-1] # return the move that was most visited
    

def testDifferenceOfPrior(player1=UCTcTest,player2=UCTcTest,nPositions=1000000):
    global diffCountG
    global countG
    global perplexity
    global countPerp   
    global priorAcc
    global countAcc  
    global MRR
    global MNC
    global crossEntropy
    global entropy
    


    print "nPositions=" + str(nPositions)
    for iter1 in [100,1000,10000,20000,100000]: #[10,1,.5,.2,.1,.05,.02,.01,.001,.0001]:
        iter2 = iter1
        diffCountG = [0,0,0,0,0,0,0,0,0,0,0,0]
        countG = 0
        perplexity=0
        countPerp=0
        priorAcc=0
        countAcc=0
        MRR=0
        MNC=0
        crossEntropy=0
        entropy = 0
        while(countG<nPositions):
            print countG
            UCTPlayNGames(player1=player1,player2=player2,iter1=iter1,iter2=iter1,nGames=1)  
        for i in range(12):
            mixing =[1000,100,10,1,.5,.2,.1,.05,.02,.01,.001,.0001][i]
            print "nIter: " + str(iter1) + ",  mixing: " + str(mixing)
        #print "Mixing value: " + str(mixing)
        
            print "The move is different " + str(diffCountG[i] / float(countG)*100) + "% of the time, nPositions= " + str(countG)
            print ""

        diffCountG = [0,0,0,0,0,0,0,0,0,0,0,0]
        countG = 0
        
        print "Accuracy of NN at "+str(iter1) +" is: " + str(priorAcc/float(countAcc)*100) + "%" + ", based on " + str(countAcc) + " positions"
        print "Mean Reciprocal Rank is: " + str(1/MRR) + ", based on " + str(countAcc) + " positions"
        print "Mean Number of Choices is: " + str(1/MNC) + ", based on " + str(countAcc) + " positions"
        print "Mean Perplexity is: " + str(perplexity) + ", based on " + str(countPerp) + " positions"
        print "Mean CrossEntropy is: " + str(crossEntropy) + ", based on " + str(countAcc) + " positions"
        print "Mean Entropy is: " + str(entropy) + ", based on " + str(countAcc) + " positions"
    
def getBoard(node):
    movesRev = []
    while node != None:
        movesRev.append(node.move)
        node = node.parentNode
    movesRev = movesRev[:-1]
    player = 2 - len(movesRev) % 2
    board = numpy.zeros((5, 5))
    for move in movesRev:
        board[move] = player
        player = 3-player
        
    #print "MOVESREV: " + str(movesRev)
    return board
 

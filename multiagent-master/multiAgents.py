# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        DESCRIPTION: Pacman eats the closet food where the distance is calculated
        using the manhattan distance between Pacman and the food. If the ghosts 
        get too close a very large negative value is returned so that Pacman 
        avoids the ghosts. The score + 1 / minimum food distance is returned. 
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # eat food 
        foodList = newFood.asList()
        minFood = float('inf')
        for food in foodList:
            minFood = min(minFood, manhattanDistance(newPos, food))
        
        # if ghost gets too close, avoid
        for ghost in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            if ghostDistance < 1:
                return -float('inf') 

        return successorGameState.getScore() + 1 / minFood

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)

    DESCRIPTION: 
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        DESCRIPTION: Returns the best action from the max or min agent.
        Where the max agent is used for pacman (agent = 0) and the min agent
        is used for the ghosts (agent >= 1).
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
            # won/lost or depth reached, return
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)

            if agent == 0:
                maximum = -float('inf')
                for agentState in gameState.getLegalActions(0):
                    utility = minimax(1, depth, gameState.generateSuccessor(agent, agentState))
                    if utility > maximum:
                        maximum = utility
                return maximum
            else:
                min = float('inf')
                for agentState in gameState.getLegalActions(agent):
                    count = 0
                    if (agent + 1) % gameState.getNumAgents() == 0:
                        count = 1
                    utility = minimax((agent + 1) % gameState.getNumAgents(), depth + count, gameState.generateSuccessor(agent, agentState))
                    if utility < min:
                        min = utility
                return min

        maximum = -float('inf')
        action = gameState
        for agentState in gameState.getLegalActions(0):
            utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum:
                maximum = utility
                action = agentState
        
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        DESCRIPTION: The agent is very similar to the minimax agent. The difference
        is adding pruning. For the maximizer, if max > beta then return the max.
        Then alpha is set to the max between alpha and max. Max is then returned.
        And vise versa is done for the minimizer.
        """
        "*** YOUR CODE HERE ***"
        def maximizer(agent, depth, gameState, a, b):
            maximum = -float('inf')
            for agentState in gameState.getLegalActions(agent):
                maximum = max(maximum, alphabetaprune(1, depth, gameState.generateSuccessor(agent, agentState), a, b))
                if maximum > b:
                    return maximum
                a = max(a, maximum)
            return maximum

        def minimizer(agent, depth, gameState, a, b):
            minimum = float('inf')
            for agentState in gameState.getLegalActions(agent):
                count = 0
                if (agent + 1) % gameState.getNumAgents() == 0:
                    count = 1
                minimum = min(minimum, alphabetaprune((agent + 1) % gameState.getNumAgents(), depth + count, gameState.generateSuccessor(agent, agentState), a, b))
                if minimum < a:
                    return minimum
                b = min(b, minimum)
            return minimum

        def alphabetaprune(agent, depth, gameState, a, b):
            # won/lost or depth reached, return
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)

            # maximize for pacman
            if agent == 0:
                return maximizer(agent, depth, gameState, a, b)
            # minimize for ghosts
            else:
                return minimizer(agent, depth, gameState, a, b)
        
        utility = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            value = alphabetaprune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if value > utility:
                utility = value
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)
        
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.

        DESCRIPTION: The agent is the same as the minimax one. The only change is 
        when calculating the expectimax. The total is added to every iteration
        and then averaged by dividing the total by the length of legal actions list.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):
            # won/lost or depth reached, return self.evaluationFunction(gameState)
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            if self.depth == depth:
                return self.evaluationFunction(gameState)
            
            # agent 0 = pacman
            # maximize
            if agent == 0:
                maximum = -float('inf')
                for agentState in gameState.getLegalActions(0):
                    utility = expectimax(1, depth, gameState.generateSuccessor(agent,agentState))
                    if utility > maximum:
                        maximum = utility
                return maximum
            # exectimax, take average for total
            else:  # agent >= 1 are the ghosts 
                total = 0
                for agentState in gameState.getLegalActions(agent):
                    count = 0
                    # if the next node is 0, then it is a new step for the pacman, depth + 1
                    if (agent+1) % gameState.getNumAgents() == 0:
                        count = 1
                    total += (expectimax((agent+1) % gameState.getNumAgents(), depth + count, gameState.generateSuccessor(agent,agentState)))
                return total/len(gameState.getLegalActions(agent))
        
        maximum = -float('inf')
        action = gameState
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0,agentState))
            if utility > maximum:
                maximum = utility
                action = agentState

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Pacman first eats the closest food pellets where the distances 
    are calculated using the manhattan distance. Pacman eats the capsules in a 
    similar fashion. The evaluation function calculates the distance to the closest 
    ghost in the same manner. If the ghost is not scared, the distance to the ghost
    is subtracted from the score returned. However, if the ghost is scared the 
    function adds the minimum ghost distance to the score. 
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()
    
    # eat food 
    foodList = newFood.asList()
    minFood = float('inf')
    minGhost = float('inf')
    minCapsules = float('inf')
    ghoststateclose = 1
    for food in foodList:
        minFood = min(minFood, manhattanDistance(newPos, food))

    # eat capsules
    for capsules in newCapsules:
        minCapsules = min(minCapsules, manhattanDistance(newPos, capsules))

    # if ghost gets too close, avoid
    for ghost in newGhostStates:
        minGhost = min(minGhost, manhattanDistance(newPos, ghost.getPosition()))
        if manhattanDistance(newPos, ghost.getPosition()) == minGhost and ghost.scaredTimer != 0:
            ghoststateclose = -1
        if (minGhost < 2 and ghost.scaredTimer == 0):
            return -float('inf') 
    
    return currentGameState.getScore() + 1 / (minFood + 1) - ghoststateclose * 2 / (minGhost + 1) + 1 / (minCapsules + 1)    # avoid divide by 0

# Abbreviation
better = betterEvaluationFunction

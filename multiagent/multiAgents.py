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

def manDist(A, B):
    """
    returns the hamiltonian distance between the two points, A and B
    """
    return abs(A[0] - B[0]) + abs(A[1] - B[1])


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def tspSoln(points):
        points = newPos + newFood
        costs = {}
        for point in points:
            costs[newPos, 0] = 0
        return None
        
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
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPosis = [ghostState.getPosition() for ghostState in newGhostStates]
        
        fardot = 0
        closedot = float('inf') if len(newFood)!= 0 else 0
        for food in newFood:
            manD = manDist(newPos, food)
            if fardot < manD:
                fardot = manD
            if closedot > manD:
                closedot = manD
        
        bestGhost = [float('inf'), 0]
        for ghost in newGhostStates:
            # Calculate the closest ghost - closer is worse than farther.
            ghostDist = manDist(newPos, ghost.getPosition())
            scareTime = ghost.scaredTimer if ghost.scaredTimer > ghostDist + 3 else float('inf')
            # Ghost score
            scareval = 10000/(scareTime + ghostDist + 1)
            distval = -(100/(ghostDist + 1)) if ghostDist < 3 else 0
            if distval < bestGhost[0]:
                bestGhost[0] = distval
            if scareval > bestGhost[1]:
                bestGhost[1] = scareval
        
        "*** YOUR CODE HERE ***"
        randVal = random.randint(0, 2)
        ghostScore = sum(bestGhost)
        numfood = 1000.0 / ((len(newFood) + 1)**2)
        fardotVal = 10.0/((fardot+ 1) ** 2) + 10
        closedotVal = 10.0/((closedot + 1) ** 2)
        value = fardotVal + ghostScore + closedotVal + successorGameState.getScore() + randVal
        #print(action, fardotVal, ghostScore, closedotVal, value)
        if len(newFood) < 0:
            raise Exception, "Negative amount of food"
            print "A MAJOR BUG"
        
        return value

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
    """

    def getValue(self, gameState, agent, depth, evalFn):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFn(gameState)

        actions = gameState.getLegalActions(agent)
        nextStates = [gameState.generateSuccessor(agent, action) for action in actions]
        nextAgent = (agent + 1) % gameState.getNumAgents()
        depth = depth - 1 if nextAgent == 0 else depth
        values = [self.getValue(state, nextAgent, depth, evalFn) for state in nextStates]
        comparator = max if (agent == 0) else min
        #print agent, values, comparator(values)
        return comparator(values)    
    
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
        """
        print self.depth
        currAgent, nextAgent = self.index, self.index + 1
        actions = gameState.getLegalActions(currAgent)
        nextStates = [(action, gameState.generateSuccessor(currAgent, action)) for action in actions]
        bestAction = None, -float('inf')
        
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            print "PROBLEM !!!!!"
            return actions[random.randint(0, len(actions) - 1)]
        
        for state in nextStates:
            value = self.getValue(state[1], nextAgent, self.depth, self.evaluationFunction)
            if value > bestAction[1]:
                bestAction = state[0], value
        return bestAction[0]
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getValue(self, gameState, agent, depth, evalFn, alphBeta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFn(gameState)

        actions = gameState.getLegalActions(agent)
        nextAgent = (agent + 1) % gameState.getNumAgents()
        depth = depth - 1 if nextAgent == 0 else depth
        
        chooser, values = max if agent == 0 else min, []
        for action in actions:
            state = gameState.generateSuccessor(agent, action)
            values.append(self.getValue(state, nextAgent, depth, evalFn, alphBeta))

            if agent == 0:
                if (alphBeta[1] < values[len(values) - 1]):
                    break
                alphBeta = (chooser(alphBeta[0], values[len(values) - 1]), alphBeta[1])
            else:
                if (alphBeta[0] > values[len(values) - 1]):
                    break                
                alphBeta = (alphBeta[0], chooser(alphBeta[1], values[len(values) - 1]))

        return chooser(values)
    
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
        """
        currAgent, nextAgent = self.index, self.index + 1
        actions = gameState.getLegalActions(currAgent)
        nextStates = [(action, gameState.generateSuccessor(currAgent, action)) for action in actions]
        bestAction = None, -float('inf')
        alphbeta = (- float('inf'), float('inf'))
        
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            print "PROBLEM !!!!!"
            return actions[random.randint(0, len(actions) - 1)]
        
        for state in nextStates:
            value = self.getValue(state[1], nextAgent, self.depth, self.evaluationFunction, alphbeta)
            alphbeta = (max(value, alphbeta[0]), alphbeta[1])
            if value > bestAction[1]:
                bestAction = state[0], value
        return bestAction[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getValue(self, gameState, agent, depth, evalFn):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return evalFn(gameState)

        actions = gameState.getLegalActions(agent)
        nextAgent = (agent + 1) % gameState.getNumAgents()
        depth = depth - 1 if nextAgent == 0 else depth
        values = []
        
        for action in actions:
            state = gameState.generateSuccessor(agent, action)
            values.append(self.getValue(state, nextAgent, depth, evalFn))

        if agent == 0:
            return max(values)
        else:
            return sum(values)/len(values)    

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        currAgent, nextAgent = self.index, self.index + 1
        actions = gameState.getLegalActions(currAgent)
        nextStates = [(action, gameState.generateSuccessor(currAgent, action)) for action in actions]
        bestAction = None, -float('inf')
        
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            raise Exception, "Bad agent definition"
            return actions[random.randint(0, len(actions) - 1)]
        
        for state in nextStates:
            value = self.getValue(state[1], nextAgent, self.depth, self.evaluationFunction)
            if value > bestAction[1]:
                bestAction = state[0], value
        return bestAction[0]
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods, capsules = currentGameState.getFood().asList(), currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPosis = [ghostState.getPosition() for ghostState in ghostStates]
        
    fardot = 0
    closedot = float('inf') if len(foods)!= 0 else 0
    for food in foods:
        manD = manDist(pos, food)
        if fardot < manD:
            fardot = manD
        if closedot > manD:
            closedot = manD
        
    bestGhost = [float('inf'), 0]
    for ghost in ghostStates:
        # Calculate the closest ghost - closer is worse than farther.
        ghostDist = manDist(pos, ghost.getPosition())
        scareTime = ghost.scaredTimer if ghost.scaredTimer > ghostDist + 3 else float('inf')
        # Ghost score
        scareval = 15/(scareTime + ghostDist + 1) 
        distval = -(10/(ghostDist + 1)) if ghostDist < 2  and scareval == 0 else 0
        if distval < bestGhost[0]:
            bestGhost[0] = distval
        if scareval > bestGhost[1]:
            bestGhost[1] = scareval
    capVal = -15 * len(capsules)
    ghostScore = sum(bestGhost)
    numfood = 100.0 / ((len(foods) + 1)**2)
    fardotVal = 10.0/((fardot+ 1) ** 2)
    closedotVal = 10.0/((closedot + 1) ** 2) + 20
    value = fardotVal + ghostScore + closedotVal + currentGameState.getScore()*2 + capVal
    if len(foods) < 0:
        raise Exception, "Negative food amount"
            
    return value
    

# Abbreviation
better = betterEvaluationFunction


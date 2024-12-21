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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


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
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        
        # closest food - more points for closer food 
        closestDistF = float('inf')
        food = newFood.asList()
        for f in food:
          dist = manhattanDistance(newPos, f)
          if dist < closestDistF:
            closestDistF = dist
          
          if closestDistF != float('inf'):
            score = score + 20 / (closestDistF + 1) # higher score for closer dist
            
          score = score - 5*len(food) # more food left = penalty = encourage pacman to eat
          
        # closest ghost and scared times
        gIndex = 0
        for g in newGhostStates:
          gDist = manhattanDistance(newPos, g.getPosition())
          scaredTime = newScaredTimes[gIndex]
          
          if scaredTime > 0:
            # more points for closer ghost
            # score = score + 20 / (gDist + 1)
            score = score + 50 / (gDist + 1)
          else:
            if gDist < 2: # high risk
              score = score - 1000 
            elif gDist < 5: # moderate risk
              score = score - 500 
            # gDist > 5 --> no risk, so no need to penalize
            
        # don't stop
        if Directions.STOP == action:
          score = score - 500
        
        # checking for walls
        canGoDirection = 0
        directionVect = {
          Directions.NORTH: (0, 1),
          Directions.SOUTH: (0, -1),
          Directions.EAST: (1, 0),
          Directions.WEST: (-1, 0)
        }        
        for d in directionVect:
          x, y = directionVect[d]
          nextPos = (newPos[0] + x, newPos[1] + y)
          
          wall = currentGameState.getWalls()
          if not wall[nextPos[0]][nextPos[1]]:
            canGoDirection = canGoDirection + 1
            
          # if theres less directions to go -> avoid
          if canGoDirection <= 1:
            score = score - 200
        
        return score

def scoreEvaluationFunction(currentGameState):
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
    def getAction(self, gameState):
        # if games over or reached max depth
        def minimax(agentI, depth, gameState):
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
          
          if agentI == 0:
            return maxVal(agentI, depth, gameState)
          else: 
            return minVal(agentI, depth, gameState)
          
        def maxVal(agentI, depth, gameState):
          value = float('-inf')
          for x in (gameState.getLegalActions(agentI)):
            successor = gameState.generateSuccessor(agentI, x)
            value = max(value, minimax(1, depth, successor)) 
          return value
        
        def minVal(agentI, depth, gameState):
          # to know whos the next agent
          value = float('inf')
          nextA = agentI + 1
          if agentI == gameState.getNumAgents() - 1: 
            # After last ghost, it's Pacman's turn
            nextA = 0 
            depth = depth + 1
          
          for x in (gameState.getLegalActions(agentI)):
            successor = gameState.generateSuccessor(agentI, x)
            value = min(value, minimax(nextA, depth, successor))
          return value
        
        # recurive
        bestScore = float('-inf')
        bestAction = None
        # to determine the best action
        for x in (gameState.getLegalActions(0)):
            successor = gameState.generateSuccessor(0, x)
            score = minimax(1, 0, successor)  
            if score > bestScore:
                bestScore = score
                bestAction = x

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBetaSearch(agentI, depth, gameState, a, b):
          if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
          
          if agentI == 0:
            return maxValue(agentI, depth, gameState, a, b)
          else: 
            return minValue(agentI, depth, gameState, a, b)
          
        def maxValue(agentI, depth, gameState, a, b):
          v = float('-inf')
              
          for x in (gameState.getLegalActions(agentI)):
            successor = gameState.generateSuccessor(agentI, x)
            v = max(v, alphaBetaSearch(1, depth, successor, a, b))
            if v > b:
              return v
            a = max(a, v)   
          return v   
        
        def minValue(agentI, depth, gameState, a, b):
          v = float('inf')
          nextAgent = agentI + 1
          if agentI == gameState.getNumAgents() - 1: # last ghost
                nextAgent = 0
                depth = depth + 1
                
          for x in (gameState.getLegalActions(agentI)):
            successor = gameState.generateSuccessor(agentI, x)
            v = min(v, alphaBetaSearch(nextAgent, depth, successor, a, b))
            if v < a:
              return v
            b = min(b, v)   
          return v   
        
        bestMove = None
        bestScore = float('-inf')
        a = float('-inf')
        b = float('inf')
      
        for move in gameState.getLegalActions(0):
          score = alphaBetaSearch(1, 0, gameState.generateSuccessor(0, move), a, b)
          if score > bestScore:
            bestScore = score
            bestMove = move
          a = max(a, bestScore)
        
        return bestMove
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      def expectimax(agentIndex, depth, gameState):
        # if game finished or depths limit reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
          return self.evaluationFunction(gameState)

        # Pacman's move
        if agentIndex == 0:
          bestScore = float('-inf')
          for x in gameState.getLegalActions(agentIndex):
              successor = gameState.generateSuccessor(agentIndex, x)
              score = expectimax(1, depth, successor)
              bestScore = max(bestScore, score)
          return bestScore

        # Ghosts move 
        else:
          nextAgent = (agentIndex + 1) % gameState.getNumAgents()
          if nextAgent == 0:
              nextDepth = depth + 1
          else:
              nextDepth = depth
              
          totalScore = 0
          for x in gameState.getLegalActions(agentIndex):
              successor = gameState.generateSuccessor(agentIndex, x)
              totalScore = totalScore + expectimax(nextAgent, nextDepth, successor)

          return totalScore / len(gameState.getLegalActions(agentIndex))

      bestScore = float('-inf')
      bestMove = None

      for x in gameState.getLegalActions(0):
          score = expectimax(1, 0, gameState.generateSuccessor(0, x))
          if score > bestScore:
              bestScore = score
              bestMove = x
      return bestMove

def betterEvaluationFunction(currentGameState):
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # distance to the nearest food
    if currentGameState.getFood().asList():
      foodDist = []
      for x in currentGameState.getFood().asList():
        distance = manhattanDistance(pacmanPos, x)
        foodDist.append(distance)

      closestFoodDist = min(foodDist)
      score = score + 10 / closestFoodDist

    # distance to ghosts 
    for x in currentGameState.getGhostStates():
        ghostDist = manhattanDistance(pacmanPos, x.getPosition())
        if x.scaredTimer > 0:
            score = score + 500 / ghostDist # go near scared ghosts
        else:
            if ghostDist < 2:
                score = score - 500 # close to an active ghost
    
    # distance to capsules
    if currentGameState.getCapsules():
      distances = []  
      for x in currentGameState.getCapsules():
          dist = manhattanDistance(pacmanPos, x) 
          distances.append(dist) 
      
      minAreaDist = min(distances)
      score = score + 100 / minAreaDist  

    # if food remaining
    score = score - 10 * len(currentGameState.getFood().asList())  
    # if areas remaining
    score = score - 100 * len(currentGameState.getCapsules())
      
    return score

# Abbreviation
better = betterEvaluationFunction


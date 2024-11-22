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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # initialize score with the game score
        score = successorGameState.getScore()

        # evaluate food
        """
        if there is food, find the distance to the closest food and add 1/distance to the score
        penalize states with more remaining food
        """
        # get food list
        foodList = newFood.asList()
        if foodList:
            # distance to closest food
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            minFoodDist = min(foodDistances)
            # add 1/distance to the score
            score += 1.0 / (minFoodDist + 1)  # we add 1 to avoid division by zero
            # penalize states with more remaining food
            score -= len(foodList) * 0.1

        # ghost evaluation
        for ghostState in newGhostStates:
            # get ghost pos
            ghostPos = ghostState.getPosition()
            # get distance
            ghostDist = manhattanDistance(newPos, ghostPos)

            # eval ghost states (scared or not)
            if ghostState.scaredTimer > 0:
                # if ghost is scared, being closer is good
                score += 2.0 / (ghostDist + 1)
            else:
                # if ghost is too close, heavily penalize
                if ghostDist < 2:
                    score -= 500
                # general penalty for being close to ghost
                score -= 2.0 / (ghostDist + 1)
        # penalise stopping
        if action == Directions.STOP:
            score -= 10
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentIndex):
            # if weve reached a terminal state (or maximum depth)
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            # get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)

            # if there are no legal actions, eval state
            if not legalActions:
                return self.evaluationFunction(state), None

            # calculate next agent index and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # init best score and action
            bestAction = None

            if agentIndex == 0:  # pacman (maximizing)
                bestScore = float('-inf')
                for action in legalActions:

                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = minimax(successor, nextDepth, nextAgent)

                    if score > bestScore:
                        bestScore = score
                        bestAction = action
            else:   # ghost (minimizing)
                bestScore = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = minimax(successor, nextDepth, nextAgent)
                    if score < bestScore:
                        bestScore = score
                        bestAction = action
            return bestScore, bestAction

        # start minimax for agend 0 (pacman)
        _, action = minimax(gameState, self.depth, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # final states
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            # get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None

            # calc next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # init best action and value
            bestAction = None
            if agentIndex == 0:  # pacman -> maximizing
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                    if score > value:
                        value = score
                        bestAction = action
                    alpha = max(alpha, value)

                    # pruning process
                    if alpha > beta:
                        break
                return value, bestAction

            else:  # ghosts -> minimizing
                value = float('inf')
                for action in legalActions:

                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                    if score < value:
                        value = score
                        bestAction = action
                    beta = min(beta, value)
                    if beta < alpha:  # pruning process
                        break
                return value, bestAction

        # start alpha beta search
        _, action = alphaBeta(gameState, self.depth, 0, float('-inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            # final states
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            # get legal actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None

            # calc next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # pacman -> maximizing
                bestScore = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextDepth, nextAgent)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return bestScore, bestAction

            else:  # ghosts -> expectation
                totalScore = 0
                # uniform probability for each action
                probability = 1.0 / len(legalActions)
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextDepth, nextAgent)
                    totalScore += probability * score
                return totalScore, None

        # start expectimax from pacmans point of view
        _, action = expectimax(gameState, self.depth, 0)
        return action


def betterEvaluationFunction(currentGameState: GameState):

    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function considers several features:
    1. Current game score
    2. Distance to closest food
    3. Number of remaining food pellets
    4. Ghost positions and states (scared vs normal)
    5. Distance to power pellets
    6. Remaining scared timer for ghosts

    Each feature is weighted appropriately and combined for a final score.
    """
    # get curent state info
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # init score w/ current game score
    score = currentGameState.getScore()

    # food eval
    foodList = food.asList()

    if foodList:
        # dist to closest food
        foodDistances = [manhattanDistance(pos, food) for food in foodList]
        minFoodDist = min(foodDistances)
        score += 10.0 / (minFoodDist + 1)  # Closer food is better
        # Penalize for remaining food
        score -= len(foodList) * 4

    # ghost eval
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)
        if ghostState.scaredTimer > 0:  # Ghost is scared
            # Reward being close to scared ghosts
            score += 200.0 / (ghostDist + 1)
            # Bonus for remaining scared time
            score += ghostState.scaredTimer * 10
        else:  # Ghost is dangerous
            if ghostDist < 2:
                # Heavy penalty for being very close to active ghost
                score -= 500
            else:
                # Smaller penalty for being near active ghost
                score -= 100.0 / (ghostDist + 1)

    # power pellet eval
    if capsules:
        # dist to closest capsule
        capsuleDistance = [manhattanDistance(pos, capsule) for capsule in capsules]
        minCapsDist = min(capsuleDistance)

        # encourage getting power pellets
        score += 50.0 / (minCapsDist + 1)

        # penalty for remaining capsules
        score -= len(capsules) * 100
    return score


# alias
better = betterEvaluationFunction

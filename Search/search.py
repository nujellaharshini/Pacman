# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    frontier = util.Stack()
    start_state = problem.getStartState()
    node = {'state': problem.getStartState(), 'path': []}
    frontier.push(node)
    visited = []
    while not frontier.isEmpty():
        node = frontier.pop()
        state = node['state']
        path = node['path']
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.append(state)
            for successor, action, cost in problem.getSuccessors(state):
                if successor not in visited:
                    next_node = {'state': successor, 'path': path + [action]}
                    frontier.push(next_node)
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    node = {'state': problem.getStartState(), 'path': []}
    if problem.isGoalState(node['state']):
        return node['path']
    frontier = util.Queue()
    frontier.push(node) 
    reached = set()
    while not frontier.isEmpty():
        node = frontier.pop()
        state = node['state']
        path = node['path']
        if state in reached:
            continue
        reached.add(state)
        if problem.isGoalState(state):
            return path
        for s, action, _ in problem.getSuccessors(state):
            if s not in reached:
                next_path = path + [action]
                frontier.push({'state': s, 'path': next_path})
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start = problem.getStartState()
    node = {'state': problem.getStartState(), 'path': [], 'cost': 0}
    frontier = util.PriorityQueue()
    frontier.push(node, 0)
    reached = {}
    reached[start] = 0
    
    while not frontier.isEmpty():
        node = frontier.pop()
        state = node['state']
        path = node['path']
        cost = node['cost']
        
        if problem.isGoalState(state):
            return path 

        for s, action, move_cost in problem.getSuccessors(state):
            new_cost = cost + move_cost
            if s not in reached or new_cost < reached[s]:
                reached[s] = new_cost
                next_node = {'state': s, 'path': path + [action], 'cost': new_cost}
                frontier.push(next_node, new_cost)
    return []
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    """
    goal = get the goal state
    x_1, y_1 = state
    x_2, y_2 = goal
    return abs(x_1 - x_2) + abs(y_1 - y_2)
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()
    node = {'state': problem.getStartState(), 'path': [], 'cost': 0}
    frontier = util.PriorityQueue()
    frontier.push(node, heuristic(start, problem))
    reached = {}
    reached[start] = 0
    
    while not frontier.isEmpty():
        node = frontier.pop()
        state = node['state']
        path = node['path']
        cost = node['cost']
        
        if problem.isGoalState(state):
            return path
        
        for s, action, move_cost in problem.getSuccessors(state):
            new_cost = cost + move_cost
            nh_cost = new_cost + heuristic(s, problem)
            
            if s not in reached or new_cost < reached[s]:
                reached[s] = new_cost
                next_node = {'state': s, 'path': path + [action], 'cost': new_cost}
                frontier.push(next_node, nh_cost)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

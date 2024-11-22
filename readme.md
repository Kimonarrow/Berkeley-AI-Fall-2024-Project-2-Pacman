# Scores a perfect score of 25/25 , the editable file was multiAgents.py
  https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/


# Pacman, now with ghosts.
# Minimax, Expectimax, Evaluation, A-B pruning etc.

Q1:

we use reciprocal distances to reward closeness to food and penalize states with more remaining food. Also, we reward closeness to scared ghosts.
When we use 1/distance, closer targets give larger values. A trait we want for these 2 options

Q2:

We start by calculating the best action for pacman, then for each ghost, and then for pacman again. We alternate between maximizing and minimizing until we reach the maximum depth or a terminal state.

Q3:

We use alpha-beta pruning to improve the minimax algorithm. We keep track of the best score and action for each node and prune the search if we find a better action for the current agent.

Q4:

We use expectimax to calculate the best action for pacman. We consider the possible outcomes of the ghosts' actions and calculate the expected utility of each action.

Q5:

We use a heuristic evaluation function that rewards closeness to food, penalizes having less food left, and rewards closeness to scared ghosts and food capsules.


## FrozenLake.py

In this project, we code policy iteration, value iteration and Q-learning from scratch. Then we generate a policy for the FrozenLake environment using a greedy algorithm on Q.

### Requirements:

Python3.6+
Numpy
Gymnasium

FrozenLake is available at https://gymnasium.farama.org/

### Modifications:

We modified FrozenLake so that the reward after falling in a hole is $-1$ instead of $1$. We registered this as a new environment named 'CustomFrozenLake-v0'.

### Results (so far):

Both value iteration and policy iteration work as expected. Q-learning still struggles in many maps with a small number of episodes. Modifying the reward function and adding a decay to epsilon helped, but if the path to the goal is long, we still have trouble.

### Ideas moving forward:

Continue tuning the hyperparameters and potentially add something to prevent converging to paths that do not lead to the goal. Potentially implement some kind of search algorithm or concepts from search algorithms like keeping track of previous state-action pairs.

import numpy as np
import gymnasium as gym 
import time
from collections import defaultdict

rng = np.random.default_rng()

"""
For Q-learning

    P (dict): From gym.core.Environment
        For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a
        list of tuples of the form [(probability, nextstate, reward, terminal),...] where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS (int): number of states in the environment
    nA (int): number of actions in the environment
    gamma (float): Discount factor. Number in range [0, 1)
"""

def pre(obs):
    #maps observations to tuples (x,d) where x and d are real numbers and d represents the direction
    return tuple(list(obs['image'][:,:,0].flatten())+[obs['direction']])

def q_learning(env, nS=50, nA=3, gamma=0.9, alpha=0.9, epsilon=1, decay = 0.95, epochs = 100, max_steps=10):
    """
    Learn Q function and policy by using Q learning for a given
    gamma, alpha, epsilon and environment

    Args:
            env, nS, nA, gamma: defined at beginning of file
            alpha(float): learning rate
            epsilon(float): for epsilon greedy algorithm
            tol (float): Terminate value iteration when
                            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            Q (np.ndarray([nS,nA])): Q function resulting from Q learning
            pi (np.ndarray([nS])): policy resulting from Q iteration

    """
    # Keep track of states
    bag = defaultdict()
    i = 0

    # Initialize Q function and policy pi
    Q = np.zeros([nS,nA])
    pi = rng.integers(low=0, high=4, size=nS)

    # Run episodes of Q-learning
    for _ in range(epochs):
        obs, _ = env.reset()
        obs = pre(obs)
        if obs in bag:
            s = bag[obs]
        else:
            bag[obs]=i
            s = i 
            i += 1
        epsilon = epsilon*decay
        #max_steps = (max_steps*decay)//1
        for _ in range(max_steps):
            coin = rng.random()
            if coin < 1 - epsilon:
                a = np.argmax(Q[s])
            else:
                a = rng.integers(nA)
            obs, r, terminated, truncated, _ = env.step(a) 
            obs = pre(obs)
            if obs in bag:
                s_ = bag[obs]
            else:
                bag[obs]=i
                s_ = i 
                i += 1
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r+gamma*np.max(Q[s_]))
            s = s_
            if truncated or terminated:
                break

    # Generate greedy policy on Q
    for s in range(nS):
        pi[s] = np.argmax(Q[s])
       
    return Q, pi

def render_single(env, pi, gamma = 0.9, max_steps=50, delay=1):
    """
    Renders a single game given environment, policy and maximum number of steps.
    
    Args: 
            env: environment
            pi: policy
            max_steps: maximum number of steps to reach goal
    Returns:
            None
    """
    R = 0
    obs, info = env.reset()
    for i in range(max_steps):
        env.render()
        time.sleep(delay)
        a = pi[obs]  
        obs, r, terminated, truncated, info = env.step(a)
        R += r*(gamma**i)

        if terminated or truncated:
            break
    env.render()
    time.sleep(delay)
    if not (terminated or truncated):
        print("Anna did not reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % R)
    env.close()

if __name__ == "__main__":
    
    #Define the environment
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode="human")

    #Find policy using Q-learning
    #nS, nA = env.observation_space.n, env.action_space.n
    Q, pi = q_learning(env, decay = 0.95, max_steps=50, epochs = 50) 

    #Render a single game
    #render_single(env, pi, delay = 0, max_steps=100)
import numpy as np
import gymnasium as gym 
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super(CustomFrozenLakeEnv, self).__init__(**kwargs)
    
    def step(self, action):
        # Perform the action
        observations, reward, terminated, truncated, info = super(CustomFrozenLakeEnv, self).step(action)
        # If the episode is done and the reward is 0, it means we fell in a hole
        if terminated and reward == 0:
            reward = -1  # Change the reward for falling in a hole to -1
        return observations, reward, terminated, truncated, info

# To use the custom environment, you need to register it with Gym
from gymnasium.envs.registration import register, registry

# Check if the environment is already registered to avoid duplication errors
if True:#'CustomFrozenLake-v0' not in registry.env_specs:
    register(
        id='CustomFrozenLake-v0',
        entry_point=CustomFrozenLakeEnv,
    )

# Now you can create and use your custom environment
#env = gym.make('CustomFrozenLake-v0')

rng = np.random.default_rng()

"""
For policy_evaluation, policy_improvement, policy_iteration and valu2wsws:

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

"""
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

gym.make('FrozenLake-v1', desc=generate_random_map(size=8))

    "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

If desc=None: map_name
If desc=None and map_name=None: desc=generate_random_map(size=8) (80% of S is frozen)
If is_slippery: P(a)=1/3, P(a+1)=1/3, P(a-1)=1/3
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """
    Evaluate the value function from a given policy.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            policy (np.array[nS]): The policy to evaluate. Maps states to actions.
            tol (float): Terminate policy evaluation when
                    max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): The value function of the given policy, where value_function[s] is
                    the value of state s.
    """

    V = np.zeros(nS)
    pi = policy    
    
    d = 1

    while d > tol:
        V_ = np.zeros(nS)
        for s in range(nS):
             a = pi[s]
             for p,s_,r,_ in P[s][a]:
                V_[s] += p*(r+gamma*V[s_])
        d = np.max(np.abs(V_ - V))
        V = V_
    
    return V

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            value_from_policy (np.ndarray): The value calculated from the policy
            policy (np.array): The previous policy

    Returns:
            new_policy (np.ndarray[nS]): An array of integers. Each integer is the optimal
            action to take in that state according to the environment dynamics and the
            given value function.
    """

    pi = np.zeros(nS, dtype="int")
    V = value_from_policy
    flag = True
    
    for s in range(nS):
    
        a_0 = pi[s]
        
        A = np.zeros(nA)
        
        for a in range(nA):
            temp = 0
            for p, s_, r, _ in P[s][a]:
                temp += p*(r+gamma*V[s_])
            A[a] = temp
        pi[s] = np.argmax(A)
            
    return pi

#0X0
# policy iteration


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

    Args:
            P, nS, nA, gamma: defined at beginning of file
            tol (float): tol parameter used in policy_evaluation()

    Returns:
            value_function (np.ndarray[nS]): value function resulting from policy iteration
            policy (np.ndarray[nS]): policy resulting from policy iteration

    Hint:
            You should call the policy_evaluation() and policy_improvement() methods to
            implement this method.
    """

    V = np.zeros(nS)
    pi = np.zeros(nS, dtype=int)
    
    flag = True
    
    while flag:
        V = policy_evaluation(P,nS,nA,pi,gamma,tol)
        pi_ = policy_improvement(P, nS, nA, V, pi, gamma)
        d = pi_ - pi
        
        if np.linalg.norm(d) < tol:
            flag = False
        pi = pi_
    
    return V, pi


#value iteration

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment

    Args:
            P, nS, nA, gamma: defined at beginning of file
            tol (float): Terminate value iteration when
                            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            value_function (np.ndarray[nS]): value function resulting from value iteration
            policy (np.ndarray[nS]): policy resulting from value iteration

    """

    V = np.zeros(nS)
    pi = np.zeros(nS, dtype=int)
    
    d = 1
    while d >= tol:
        V_ = np.zeros(nS)
        for s in range(nS):
            A = np.zeros(nA)
            for a in range(nA):
                for p,s_,r,_ in P[s][a]:
                    A[a] += p*(r+gamma*V[s_])
            V_[s] = max(A)
        d = np.linalg.norm(V_ - V)
        V = V_
    
    for s in range(nS):
        A = np.zeros(nA)
        for a in range(nA):
            for p,s_,r,_ in P[s][a]:
                A[a] += p*(r+gamma*V[s_])
            Amax = np.argmax(A)
        pi[s] = Amax
       
    return V, pi

def q_learning(env, nS, nA, gamma=0.9, alpha=0.9, epsilon=1, decay = 0.95, epochs = 100, max_steps=10):
    """
    Learn Q function and policy by using Q learning for a given
    gamma, alpha, epsilon and environment

    Args:
            env, nS, nA, gamma: defined at beginning of file
            alpha: learning rate
            epsilon: for epsilon greedy algorithm
            tol (float): Terminate value iteration when
                            max |value_function(s) - prev_value_function(s)| < tol

    Returns:
            Q (np.ndarray([nS,nA])): Q function resulting from Q learning
            pi (np.ndarray([nS])): policy resulting from Q iteration

    """
    Q = np.zeros([nS,nA])
    pi = rng.integers(low=0, high=4, size=nS)

    for _ in range(epochs):
        s, _ = env.reset()
        epsilon = epsilon*decay
        for _ in range(max_steps):
            coin = rng.random()
            if coin < 1 - epsilon:
                a = np.argmax(Q[s])
            else:
                a = rng.integers(4)
            s_, r, terminated, truncated, _ = env.step(a) 
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r+gamma*np.max(Q[s_]))
            s = s_
            if truncated or terminated:
                break

    for s in range(nS):
        pi[s] = np.argmax(Q[s])
       
    return Q, pi

def render_single(env, pi, max_steps=50, delay=1):
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
    for _ in range(max_steps):
        env.render()
        time.sleep(delay)
        a = pi[obs]  
        obs, r, terminated, truncated, info = env.step(a)
        R += r

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
    #Create custom map
    d = ['HHHHH',
         'HSFFG',
         'HHHHH']
    
    #Define the environment
    env = gym.make('CustomFrozenLake-v0', desc=d, map_name="4x4", is_slippery=False)#, render_mode="human")

    #Find policy using policy iteration
    #V_pi, p_pi = policy_iteration(env.P, env.observation_space.n, env.action_space.n)
    #render_single(env, p_pi)

    #Find policy using value iteration
    #V_vi, p_vi = value_iteration(env.P, env.observation_space.n, env.action_space.n)
    #render_single(env, p_vi)

    #Find policy using Q-learning
    nS, nA = env.observation_space.n, env.action_space.n
    Q, pi = q_learning(env, env.observation_space.n, env.action_space.n, decay = 0.99, max_steps=50, epochs = 1000) 

    #Print the final policy generated by Q
    print(pi.reshape(3,5))

    #Render a single game
    env = gym.make('CustomFrozenLake-v0', desc=d, map_name="4x4", is_slippery=False, render_mode="human") 
    render_single(env, pi, delay = 1, max_steps=6)

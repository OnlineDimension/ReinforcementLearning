import numpy as np
import gymnasium as gym 
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

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

def render_single(env, pi, max_steps=1000):

	R = 0
	obs, info = env.reset()
	for _ in range(max_steps):
		env.render()
		#time.sleep(0.25)
		a = pi[obs]  
		obs, r, terminated, truncated, info = env.step(a)
		R += r

		if terminated or truncated:
			break
	env.render()
	if not (terminated or truncated):
		print("Anna did not reach a terminal state in {} steps.".format(max_steps))
	else:
		print("Episode reward: %f" % R)

if __name__ == "__main__":
	
    """
	d = ['SFHFG',
		 'FFHFF',
		 'FFHFF',
		 'FFHFF',
		 'FFFFF']
	"""

    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=9), is_slippery=True, render_mode="human")


	#V_pi, p_pi = policy_iteration(env.P, env.observation_space.n, env.action_space.n)
	#render_single(env, p_pi)

    V_vi, p_vi = value_iteration(env.P, env.observation_space.n, env.action_space.n)
    render_single(env, p_vi)
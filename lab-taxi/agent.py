import numpy as np
from collections import defaultdict
from enum import Enum, auto

class AlgorithmType(Enum):
    MonteCarlo = auto()
    Sarsa = auto()
    QLearning = auto()
    ExpectedSarsa = auto()

class Agent:

    def __init__(self, nA=6, type=AlgorithmType.ExpectedSarsa):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.start_epsilon = 0.8
        self.epsilon = self.start_epsilon
        self.epsilon_decay = lambda e: e*self.start_epsilon
        self.alpha = 0.1
        self.gamma = 0.8
        self.episode_counter = 0
        self.algorithm_type = type

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        greedy_action = np.argmax(self.Q[state])
        return greedy_action if np.random.uniform() > self.epsilon else np.random.randint(0, self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if self.algorithm_type == AlgorithmType.QLearning:
            # Q-learning
            self.Q[state][action] += self.alpha * (reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action])

        elif self.algorithm_type == AlgorithmType.ExpectedSarsa:
            ## Expected Sarsa
            pi = np.ones(self.nA) * self.epsilon / self.nA
            pi[np.argmax(self.Q[state])] += 1 - self.epsilon
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.sum(pi * self.Q[next_state]) - self.Q[state][action])
        else:
            raise ValueError("Algorithm type %s is not supported" % (self.algorithm_type))


        if done:
            self.episode_counter += 1
            self.epsilon = self.epsilon_decay(self.epsilon)
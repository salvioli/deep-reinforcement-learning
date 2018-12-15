from agent import Agent, AlgorithmType
from monitor import interact
import gym
import numpy as np
from collections import defaultdict
import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


env = gym.make('Taxi-v2')

num_episodes = 10000
alphas = np.arange(0.1, 1, 0.1)
epsilons = np.arange(0.1, 1, 0.1)
algo_types = [AlgorithmType.QLearning, AlgorithmType.ExpectedSarsa]
# experiment_tag = 'alpha-.1-1-.1-eps-.1-1-.1'

# avg_rewards = defaultdict(dict)
# best_avg_rewards = defaultdict(dict)

# for t in algo_types:
#     for i, a in enumerate(alphas):
#         for e in epsilons:
#             avg_rewards[a][e], best_avg_rewards[a][e] = interact(env, Agent(algorithm_type=t, epsilon=e, alpha=a), num_episodes=num_episodes)
#             if i == len(alphas)-1:
#                 save_obj(avg_rewards, 'avg_rewards-'+ experiment_tag + '-' + t.name)
#                 save_obj(best_avg_rewards, 'best_avg_rewards-' + experiment_tag + '-'  + t.name)

avg_rewards, best_avg_rewards = interact(env, Agent(), num_episodes=num_episodes)
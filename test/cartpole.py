import random

import gym
import numpy as np

from agents import DQNAgent

env = gym.make("CartPole-v0")

print("Observation space: {}".format(env.observation_space))
print("Action space: {}".format(env.action_space))

nb_actions = env.action_space.n
observation_shape = env.observation_space.shape

agent = DQNAgent(action_space=env.action_space, observation_space=env.observation_space)

min_e_greedy = 0.05
e_greedy = 1

print("Start training~")
for episode in range(100):
    episode_rewards = 0
    observation = env.reset()
    for step in range(200):
        if random.random() < max(e_greedy, min_e_greedy):
            action = random.choice(range(nb_actions))
        else:
            q_values = agent.forward(observation)
            action = np.argmax(q_values)
        
        next_observation, reward, terminal, _ = env.step(action)
        
        agent.backward(observation, action, reward, terminal, next_observation)
        episode_rewards += reward
        
        observation = next_observation
        if terminal:
            break
    
    e_greedy *= 0.9
    
    print("Episode {}: {}".format(episode, episode_rewards))

print("Start testing~")
for episode in range(20):
    episode_rewards = 0
    observation = env.reset()
    for step in range(200):
        q_values = agent.forward(observation)
        action = np.argmax(q_values)
        
        next_observation, reward, terminal, _ = env.step(action)
        
        episode_rewards += reward
        
        observation = next_observation
        if terminal:
            break
        env.render()
    
    print("Episode {}: {}".format(episode, episode_rewards))

env.close()
import os

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from agents import *
from core.utils import *

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

env = NormalizedWrapper(gym.make("Pendulum-v0"))

print("Observation space: {}".format(env.observation_space.shape))
print("Action space: {}".format(env.action_space.shape))

nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape

algo_name = "SAC"

if algo_name == "DDPG":
    agent = DDPGAgent(env.action_space, env.observation_space, nb_steps_warmup=2000)
elif algo_name == "TD3":
    agent = TD3Agent(env.action_space, env.observation_space, nb_steps_warmup=2000)
elif algo_name == "SAC":
    agent = SACAgent(env.action_space, env.observation_space, nb_steps_warmup=2000)
else:
    agent = None
    print("Undefined algorithms!!")
    exit(1)

print("Start training~")
for episode in range(300):
    episode_rewards = 0
    observation = env.reset()
    observation = observation.reshape(observation_shape).astype(np.double)
    
    for step in range(200):
        action = agent.forward(observation)
        
        next_observation, reward, terminal, _ = env.step(action)
        
        next_observation = next_observation.reshape(observation_shape).astype(np.double)
        reward = reward.astype(np.double)
        
        agent.backward(observation, action, reward, terminal, next_observation)
        episode_rewards += reward
        
        observation = next_observation
        
        if terminal:
            break
    
    print("Episode {}: {}".format(episode, episode_rewards))

agent.training = False
print("Start testing~")
for episode in range(20):
    episode_rewards = 0
    observation = env.reset()
    observation = observation.reshape(observation_shape)
    
    for step in range(200):
        env.render()
        action = agent.forward(observation)
        
        next_observation, reward, terminal, _ = env.step(action)
        next_observation = next_observation.reshape(observation_shape)
        
        episode_rewards += reward
        observation = next_observation
        
        if terminal:
            break
    
    print("Episode {}: {}".format(episode, episode_rewards))

env.close()

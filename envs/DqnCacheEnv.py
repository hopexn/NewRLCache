import numpy as np
import tensorflow as tf

from core.Memory import Memory
from envs.CacheEnv import CacheEnv


class DqnCacheEnv(CacheEnv):
    def __init__(self, capacity, request_path, top_k, time_slot_length,
                 gamma=0.99, memory_size=1000, target_model_update=10
                 ):
        super().__init__(capacity, request_path, top_k, time_slot_length)
        self.nb_actions = capacity + top_k
        self.observation_shape = (self.nb_actions,)
        
        # DQN参数
        self.gamma = gamma
        self.memory_size = memory_size
        self.target_model_update = target_model_update
        
        # ReplayBuffer
        self.memory = Memory(capacity=memory_size,
                             observation_shape=self.observation_shape,
                             action_shape=(self.nb_actions,))
        
        # 创建DQN网络
        self.q_net = self._build_q_net()
        self.target_q_net = self._build_q_net()
        self.target_q_net.set_weights(self.q_net.get_weights())
        
        self.step_count = 0
        self.last_observation = None
        self.last_action = None
        self.eps = 1
        self.eps_decay = 0.95
        self.min_eps = 0.05
    
    def _build_q_net(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.observation_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.nb_actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)
        return model
    
    def _get_new_cache_content(self, cache_content, top_k_missed_videos, hit_rate):
        candidates = np.concatenate([cache_content, top_k_missed_videos])
        observation = self.loader.get_frequencies(candidates)
        reward = hit_rate
        
        if self.last_observation is not None and self.last_action is not None:
            self.backward(self.last_observation, self.last_action, reward, False, observation)
        
        action = self.forward(observation)
        new_cache_content = candidates[action]
        
        self.last_action = action
        self.last_observation = observation
        
        return new_cache_content
    
    def forward(self, observation):
        self.step_count += 1
        action_mask = np.zeros_like(observation, dtype=np.bool)
        if np.random.random() < max(self.min_eps, self.eps):
            action = np.random.choice(np.arange(self.capacity + self.top_k), self.capacity, replace=False)
        else:
            self.eps *= self.eps_decay
            if observation.ndim == 1:
                observation = np.expand_dims(observation, axis=0)
            q_values = self.q_net.predict(observation).squeeze(0)
            action = np.argpartition(q_values, -self.capacity)[-self.capacity:]
        
        action_mask[action] = True
        return action_mask
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        q_values = self.q_net.predict(observations)
        actions = actions.astype(np.bool)
        
        target_q_values = self.target_q_net.predict(next_observations)
        new_q_values = rewards + self.gamma * target_q_values * (~terminals)
        q_values[actions] = new_q_values[actions]
        
        self.q_net.fit(observations, q_values, verbose=0)
        
        if self.step_count % self.target_model_update == 0:
            self.target_q_net.set_weights(self.q_net.get_weights())

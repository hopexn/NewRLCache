import numpy as np
import tensorflow as tf

from agents.Agent import Agent
from core.Memory import Memory


class DQNAgent(Agent):
    
    def __init__(self,
                 action_space,
                 observation_space,
                 gamma=0.99,
                 target_model_update=100,
                 memory_size=10000):
        super().__init__()
        self.gamma = gamma
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.nb_actions = action_space.n
        self.observation_shape = observation_space.shape
        self.target_model_update = target_model_update
        
        self.memory = Memory(capacity=memory_size, action_shape=(1,),
                             observation_shape=self.observation_shape)
        
        self.model = self._build_network()
        self.target_model = self._build_network()
        self.target_model.set_weights(self.model.get_weights())
        
        self.update_count = 0
    
    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.observation_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.nb_actions, activation='linear')
        ])
        model.compile(optimizer='adam', metrics=['mse'], loss=tf.keras.losses.mean_squared_error)
        return model
    
    def forward(self, observation):
        observation = np.expand_dims(observation, axis=0)
        q_values = self.model.predict(observation)
        return q_values
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        actions = tf.keras.utils.to_categorical(actions, num_classes=self.nb_actions).astype(np.bool)
        
        q_values = self.model.predict(observations)
        
        target_q_values = np.max(self.target_model.predict(next_observations), axis=1, keepdims=True)
        q_values[actions, np.newaxis] = rewards + self.gamma * target_q_values * (~terminals)
        
        self.model.fit(observations, q_values, verbose=0)
        
        self.update_count += 1
        if self.update_count % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())

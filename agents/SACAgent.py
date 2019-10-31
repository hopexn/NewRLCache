import numpy as np
import tensorflow as tf

from agents.Agent import Agent
from core.Memory import Memory
from utils.utils import *


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.math.exp(log_std) + 1e-6)) ** 2 + 2 * log_std + np.math.log(2 * np.pi))
    return tf.math.reduce_sum(pre_sum, axis=1)


class SACAgent(Agent):
    
    def __init__(self,
                 action_space,
                 observation_space,
                 gamma=0.99,
                 nb_steps_warmup=2000,
                 alpha=0.2,
                 polyak=0.995,
                 lr=3e-4,
                 log_std_min=-20,
                 log_std_max=2,
                 memory_size=10000
                 ):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.nb_steps_warmup = nb_steps_warmup
        self.lr = lr
        self.step_count = 0
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_entropy = -self.action_space.shape[0]
        
        self.memory = Memory(capacity=memory_size,
                             observation_shape=observation_space.shape,
                             action_shape=action_space.shape)
        
        self.value_net = self._build_value_network()
        self.target_value_net = self._build_value_network()
        self.target_value_net.set_weights(self.value_net.get_weights())
        
        self.soft_q_net1 = self._build_soft_q_network()
        self.soft_q_net2 = self._build_soft_q_network()
        
        self.policy_net = self._build_policy_network()
    
    def forward(self, observation):
        self.step_count += 1
        
        if self.step_count < self.nb_steps_warmup:
            return self.action_space.sample()
        else:
            if observation.ndim == 1:
                observation = np.expand_dims(observation, axis=0)
            mean, log_std = self.policy_net.predict(observation)
            
            std = tf.math.exp(log_std)
            action = mean + tf.random.normal(tf.shape(mean)) * std
            return action
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        if self.step_count >= self.nb_steps_warmup:
            self._update()
            
            new_target_weights = polyak_averaging(
                self.value_net.get_weights(),
                self.target_value_net.get_weights(),
                self.polyak
            )
            self.target_value_net.set_weights(new_target_weights)
    
    def _update(self):
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        target_q_value = rewards + self.gamma * self.target_value_net.predict(next_observations)
        
        soft_actions, log_probs = self.evaluate(observations)
        
        soft_q_value1 = self.soft_q_net1.predict([observations, soft_actions])
        soft_q_value2 = self.soft_q_net2.predict([observations, soft_actions])
        
        target_value = tf.minimum(soft_q_value1, soft_q_value2) - self.alpha * log_probs
        
        # Update soft Q network
        self.soft_q_net1.fit([observations, actions], target_q_value, verbose=0)
        self.soft_q_net2.fit([observations, actions], target_q_value, verbose=0)
        
        # Update value network
        self.value_net.fit(observations, target_value, verbose=0)
        
        # Update policy network
        with tf.GradientTape() as tape:
            tape.watch(self.policy_net.trainable_weights)
            
            soft_actions, log_probs = self.evaluate(observations)
            
            soft_q_value = self.soft_q_net1([observations, soft_actions])
            
            loss = -tf.reduce_mean(soft_q_value - self.alpha * log_probs)
        
        actor_grads = tape.gradient(loss, self.policy_net.trainable_weights)
        self.policy_net.optimizer.apply_gradients(zip(actor_grads, self.policy_net.trainable_weights))
    
    @tf.function
    def evaluate(self, observations):
        mean, log_std = self.policy_net(observations)
        
        std = tf.math.exp(log_std)
        z = mean + tf.random.normal(tf.shape(mean)) * std
        action = tf.math.tanh(z)
        log_prob = gaussian_likelihood(z, mean, log_std)
        log_prob -= tf.math.reduce_sum(tf.math.log(1 - action ** 2 + 1e-6), axis=1)
        
        action = tf.cast(action, dtype=tf.float64)
        
        return action, log_prob
    
    def _build_value_network(self):
        observation_shape = self.observation_space.shape
        
        layers = tf.keras.layers
        
        model = tf.keras.models.Sequential([
            layers.Flatten(input_shape=observation_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        
        return model
    
    def _build_soft_q_network(self):
        observation_shape = self.observation_space.shape
        nb_actions = self.action_space.shape[0]
        
        layers = tf.keras.layers
        observation_tensor = layers.Input(shape=observation_shape)
        action_tensor = layers.Input(shape=(nb_actions,))
        
        observation_tensor_flattened = layers.Flatten()(observation_tensor)
        y = layers.Concatenate()([observation_tensor_flattened, action_tensor])
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dense(1)(y)
        
        model = tf.keras.models.Model(inputs=[observation_tensor, action_tensor], outputs=y)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        
        return model
    
    def _build_policy_network(self):
        observation_shape = self.observation_space.shape
        nb_actions = self.action_space.shape[0]
        
        layers = tf.keras.layers
        observation_tensor = layers.Input(shape=observation_shape)
        y = layers.Flatten()(observation_tensor)
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dense(32, activation='relu')(y)
        
        mean = layers.Dense(nb_actions, activation='tanh')(y)
        log_std = layers.Dense(nb_actions, activation='tanh')(y)
        
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        model = tf.keras.models.Model(inputs=observation_tensor, outputs=[mean, log_std])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.lr))
        
        return model

import tensorflow as tf
from rl.random import OrnsteinUhlenbeckProcess

from agents.Agent import Agent
from core.Memory import Memory
from utils.utils import *


class DDPGAgent(Agent):
    def __init__(self,
                 action_space,
                 observation_space,
                 gamma=0.99,
                 nb_steps_warmup=2000,
                 training=True,
                 polyak=0.99,
                 memory_size=10000
                 ):
        super().__init__()
        self.gamma = gamma
        self.polyak = polyak
        
        self.action_space = action_space
        self.nb_actions = action_space.shape[0]
        self.observation_shape = observation_space.shape
        self.nb_steps_warmup = nb_steps_warmup
        self.training = training
        
        self.memory = Memory(capacity=memory_size,
                             observation_shape=self.observation_shape,
                             action_shape=self.action_space.shape)
        
        self.actor_model, self.critic_model = self._build_network()
        self.target_actor_model, self.target_critic_model = self._build_network()
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        
        self.step_count = 0
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=0.15, mu=0., sigma=0.3)
    
    def _build_network(self):
        action_tensor = tf.keras.layers.Input(shape=(self.nb_actions,), dtype=tf.float64)
        observation_tensor = tf.keras.layers.Input(shape=self.observation_shape, dtype=tf.float64)
        
        # 创建Actor模型
        y = tf.keras.layers.Flatten()(observation_tensor)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(self.nb_actions, activation='tanh')(y)
        
        actor_model = tf.keras.Model(inputs=observation_tensor, outputs=y)
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='mse')
        
        # 创建Critic模型
        y = tf.keras.layers.Concatenate()([observation_tensor, action_tensor])
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(1, activation='linear')(y)
        
        critic_model = tf.keras.Model(inputs=[observation_tensor, action_tensor], outputs=y)
        critic_model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='mse')
        
        return actor_model, critic_model
    
    def forward(self, observation):
        self.step_count += 1
        
        if self.step_count < self.nb_steps_warmup:
            return self.action_space.sample()
        else:
            observation = np.expand_dims(observation, axis=0)
            action = self.actor_model.predict(observation)
            action = action.reshape(self.nb_actions)
            if self.training:
                action = action + self.random_process.sample()
            return action
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        if self.step_count < self.nb_steps_warmup:
            return
        else:
            self._update()
    
    def _update(self):
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch()
        
        self._update_critic(observations, actions, rewards, terminals, next_observations)
        self._update_actor(observations)
        
        # 更新critic的target网络
        new_target_critic_weights_list = polyak_averaging(
            self.critic_model.get_weights(), self.target_critic_model.get_weights(), self.polyak)
        self.target_critic_model.set_weights(new_target_critic_weights_list)
        
        # 更新actor的target网络
        new_target_actor_weights_list = polyak_averaging(
            self.actor_model.get_weights(), self.target_actor_model.get_weights(), self.polyak)
        self.target_actor_model.set_weights(new_target_actor_weights_list)
    
    def polyak_averaging(self, weights_list, target_weights_list):
        new_target_weights_list = []
        for weights, target_weights in zip(weights_list, target_weights_list):
            new_target_weights = self.polyak * target_weights + (1 - self.polyak) * weights
            new_target_weights_list.append(new_target_weights)
        return new_target_weights_list
    
    def _update_critic(self, observations, actions, rewards, terminals, next_observations):
        q_values_next = self.target_critic_model([next_observations, self.actor_model(next_observations)])
        target_q_values = rewards + self.gamma * q_values_next
        self.critic_model.fit([observations, actions], target_q_values, verbose=0)
    
    @tf.function
    def _update_actor(self, observations):
        with tf.GradientTape() as tape:
            tape.watch(self.actor_model.trainable_weights)
            q_values = self.target_critic_model([observations, self.actor_model(observations)])
            loss = -tf.reduce_mean(q_values)
        
        actor_grads = tape.gradient(loss, self.actor_model.trainable_weights)
        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_weights))

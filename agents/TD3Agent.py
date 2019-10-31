import numpy as np
import tensorflow as tf

from agents.Agent import Agent
from core.Memory import Memory
from utils.utils import *


class TD3Agent(Agent):
    def __init__(self,
                 action_space,
                 observation_space,
                 gamma=0.99,
                 nb_steps_warmup=2000,
                 sigma=0.3,
                 polyak=0.995,
                 pi_lr=0.001,
                 q_lr=0.001,
                 batch_size=100,
                 action_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 memory_size=10000,
                 training=True):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.action_space = action_space
        self.nb_actions = action_space.shape[0]
        self.observation_shape = observation_space.shape
        self.nb_steps_warmup = nb_steps_warmup
        self.training = training
        
        self.memory = Memory(capacity=memory_size,
                             observation_shape=self.observation_shape,
                             action_shape=self.action_space.shape)
        
        self.actor_model, self.critic_model1, self.critic_model2 = self._build_network()
        
        self.target_actor_model, self.target_critic_model1, self.target_critic_model2 = self._build_network()
        
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model1.set_weights(self.critic_model1.get_weights())
        self.target_critic_model2.set_weights(self.critic_model2.get_weights())
        
        self.step_count = 0
    
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
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.pi_lr), loss='mse')
        
        # 创建Critic1模型
        critic_model1 = self._build_critic_network(observation_tensor, action_tensor)
        # 创建Critic2模型
        critic_model2 = self._build_critic_network(observation_tensor, action_tensor)
        
        return actor_model, critic_model1, critic_model2
    
    def _build_critic_network(self, observation_tensor, action_tensor):
        y = tf.keras.layers.Concatenate()([observation_tensor, action_tensor])
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(32, activation='relu')(y)
        y = tf.keras.layers.Dense(1, activation='linear')(y)
        
        critic_model = tf.keras.Model(inputs=[observation_tensor, action_tensor], outputs=y)
        critic_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.q_lr), loss='mse')
        return critic_model
    
    def forward(self, observation):
        self.step_count += 1
        
        if self.step_count < self.nb_steps_warmup:
            return self.action_space.sample()
        else:
            observation = np.expand_dims(observation, axis=0)
            action = self.actor_model.predict(observation)
            action = action.reshape(self.nb_actions)
            if self.training:
                action = action + np.clip(np.random.normal(0.0, self.action_noise, self.nb_actions),
                                          -self.noise_clip, self.noise_clip)
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
        
        if self.step_count % self.policy_delay == 0:
            # 更新critic的target网络
            new_target_critic_weights_list = polyak_averaging(
                self.critic_model1.get_weights(), self.target_critic_model1.get_weights(), self.polyak)
            self.target_critic_model1.set_weights(new_target_critic_weights_list)
            new_target_critic_weights_list = polyak_averaging(
                self.critic_model2.get_weights(), self.target_critic_model2.get_weights(), self.polyak)
            self.target_critic_model2.set_weights(new_target_critic_weights_list)
            
            # 更新actor的target网络
            new_target_actor_weights_list = polyak_averaging(
                self.actor_model.get_weights(), self.target_actor_model.get_weights(), self.polyak)
            self.target_actor_model.set_weights(new_target_actor_weights_list)
    
    def _update_critic(self, observations, actions, rewards, terminals, next_observations):
        batch_size = observations.shape[0]
        
        q_values_next1 = self.target_critic_model1([next_observations, self.actor_model(next_observations)])
        target1_noise = tf.clip_by_value(
            tf.random.normal(mean=0.0, stddev=self.target_noise, shape=(batch_size, 1), dtype=tf.float64),
            -self.noise_clip, self.noise_clip
        )
        target_q_values1 = rewards + self.gamma * q_values_next1 + target1_noise
        q_values_next2 = self.target_critic_model2([next_observations, self.actor_model(next_observations)])
        target2_noise = tf.clip_by_value(
            tf.random.normal(mean=0.0, stddev=self.target_noise, shape=(batch_size, 1), dtype=tf.float64),
            -self.noise_clip, self.noise_clip
        )
        target_q_values2 = rewards + self.gamma * q_values_next2 + target2_noise
        
        target_q_values = tf.minimum(target_q_values1, target_q_values2)
        
        self.critic_model1.fit([observations, actions], target_q_values, verbose=0)
        self.critic_model2.fit([observations, actions], target_q_values, verbose=0)
    
    @tf.function
    def _update_actor(self, observations):
        with tf.GradientTape() as tape:
            tape.watch(self.actor_model.trainable_weights)
            q_values = self.target_critic_model1([observations, self.actor_model(observations)])
            loss = -tf.reduce_mean(q_values)
        
        actor_grads = tape.gradient(loss, self.actor_model.trainable_weights)
        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_weights))

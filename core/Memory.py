import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True)
def sample_indices(batch_size, sample_range):
    return np.random.choice(np.arange(sample_range), batch_size, replace=False)


class Memory:
    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        self.observation_memory = np.zeros(
            shape=(capacity,) + observation_shape)
        self.next_observation_memory = np.zeros(
            shape=(capacity,) + observation_shape)
        self.action_memory = np.zeros(
            shape=(capacity,) + action_shape)
        self.reward_memory = np.zeros(shape=(self.capacity, 1))
        self.terminal_memory = np.zeros(shape=(self.capacity, 1), dtype=np.bool)
        
        self.write_count = 0
    
    def __len__(self):
        return min(self.write_count, self.capacity)
    
    def store_transition(self, observation, action, reward, terminal, next_observation):
        next_position = self.write_count % self.capacity
        self.write_count += 1
        
        self.observation_memory[next_position] = observation
        self.next_observation_memory[next_position] = next_observation
        self.reward_memory[next_position] = reward
        self.terminal_memory[next_position] = terminal
        self.action_memory[next_position] = action
    
    def sample_batch(self, batch_size=32):
        batch_size = min(batch_size, self.__len__())
        sample_range = min(self.write_count, self.capacity)
        indices = sample_indices(batch_size, sample_range)
        
        observations = self.observation_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        terminals = self.terminal_memory[indices]
        next_observations = self.next_observation_memory[indices]
        
        return observations, actions, rewards, terminals, next_observations

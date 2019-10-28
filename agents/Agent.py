from gym.spaces import Space


class Agent:
    def __init__(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space
    
    def forward(self, observation):
        raise NotImplementedError
    
    def backward(self, observation, action, reward, terminal, next_observation):
        raise NotImplementedError

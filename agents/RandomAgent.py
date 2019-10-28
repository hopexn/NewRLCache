from agents.Agent import Agent


class RandomAgent(Agent):
    def forward(self, observation):
        return self.action_space.sample()
    
    def backward(self, observation, action, reward, terminal, next_observation):
        pass

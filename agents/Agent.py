class Agent:
    def __init__(self):
        self.action = None

    def forward(self, observation):
        raise NotImplementedError

    def backward(self, observation, action, reward, terminal, next_observation):
        raise NotImplementedError
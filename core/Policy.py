import numpy as np


class Policy(object):
    def select_action(self, **kwargs):
        raise NotImplementedError()


class SoftmaxPolicy(Policy):
    def select_action(self, nb_actions, probs):
        action = np.random.choice(range(nb_actions), p=probs)
        return action


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        
        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action


class GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy(Policy):
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        
        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action


class MaxBoltzmannQPolicy(Policy):
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        
        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(q_values)
        return action

import gym
import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=True)
def polyak_sum(weights, target_weights, polyak):
    return target_weights * polyak + weights * (1 - polyak)


def polyak_averaging(weights_list, target_weights_list, polyak):
    assert len(weights_list) == len(target_weights_list)
    nb_layers = len(weights_list)
    
    new_target_weights_list = []
    for idx in range(nb_layers):
        new_target_weights = polyak_sum(weights_list[idx], target_weights_list[idx], polyak)
        new_target_weights_list.append(new_target_weights)
    
    return new_target_weights_list


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


class NormalizedWrapper(gym.ActionWrapper):
    def action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        new_action = low + 0.5 * (high - low) * (action + 1)
        new_action = np.clip(new_action, low, high)
        return new_action
    
    def reverse_action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        origin_action = -1 + 2 * (action - low) / (high - low)
        origin_action = np.clip(origin_action, low, high)
        return origin_action

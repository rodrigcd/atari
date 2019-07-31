import tensorflow as tf
import numpy as np


class BaseAgent(object):

    def __init__(self, **kwargs):
        self.gamma = kwargs["gamma"]
        self.memory_size = kwargs["memory_size"]
        self.batch_size = kwargs["batch_size"]
        self.training_frequency = kwargs["training_frequency"]
        self.target_network_update_freq = kwargs["target_network_update_freq"]
        self.model_persistence_update_freq = kwargs["model_persistence_update_freq"]
        self.replay_start_size = kwargs["replay_start_size"]
        self.exploration_max = kwargs["exploration_max"]
        self.exploration_min = kwargs["exploration_min"]
        self.exploration_steps = kwargs["exploration_steps"]
        self.exploration_decay = kwargs["exploration_decay"]

    def move(self, state):
        return np.random.randint(low=0, high=4, size=1)

    def remember(self, current_state, action, reward, next_state, terminal):
        pass

    def step_update(self, total_steps):
        pass

    def save_run(self, score, step, run):
        pass
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
import tensorflow as tf
import numpy as np

steering_actions = np.zeros((20, 3))
steering_actions[:, 0] = np.linspace(-1.0, 1.0, 20)
gas_actions = np.zeros((10, 3))
gas_actions[:, 1] = np.linspace(0, 1.0, 10)
brake_actions = np.zeros((10, 3))
brake_actions[:, 2] = np.linspace(0, 1.0, 10)
actions = np.vstack([steering_actions, gas_actions, brake_actions])

class RaceCarEnvProcessor(Processor):

    def __init__(self, actions):
        self.actions = actions

    def process_observation(self, observation: np.ndarray):
        '''
        :param observation: numpy array of shape (96, 96, 3) return by the environment
        :return: 96 x 96 grayscale image
        '''
        rgb_weights = [0.2989, 0.5870, 0.1140]
        return observation.dot(rgb_weights)

    def process_action(self, action):
        return self.actions[action]


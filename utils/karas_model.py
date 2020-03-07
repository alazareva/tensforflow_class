import argparse

import gym
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
from tensorflow.keras.layers import Dense, Activation, Flatten, \
    Convolution2D, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

INPUT_SHAPE = (96, 96)
WINDOW_LENGTH = 10
MEMORY_LIMIT = 1000000
WARMUP_STEPS = 50000
TARGET_MODEL_UPDATE = 10000
LEARNING_RATE = .00025
TRAIN_INTERVAL = 4
STEPS = 1750000
EVALUATION_EPISODES = 5


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

def construct_action_space(n_steering, n_gas, n_brake):
    steering_actions = np.zeros((n_steering, 3))
    steering_actions[:, 0] = np.linspace(-1.0, 1.0, n_steering)
    gas_actions = np.zeros((n_gas, 3))
    gas_actions[:, 1] = np.linspace(0, 1.0, n_gas)
    brake_actions = np.zeros((n_brake, 3))
    brake_actions[:, 2] = np.linspace(0, 1.0, n_brake)
    return  np.vstack([steering_actions, gas_actions, brake_actions])


def construct_model(window_length, n_actions):
    model = Sequential()
    # (width, height, channels)
    input_shape = (window_length,) + INPUT_SHAPE
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


# python utils/karas_model.py --steps=100
# python utils/karas_model.py --mode=test --evaluation_episodes=2

if __name__=="__main__":

    env_name = 'CarRacing-v0'
    env = gym.make(env_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--window_length', type=int, default=WINDOW_LENGTH)
    parser.add_argument('--memory_limit', type=int, default=MEMORY_LIMIT)
    parser.add_argument('--warmup_steps', type=int, default=WARMUP_STEPS)
    parser.add_argument('--target_model_update', type=int, default=TARGET_MODEL_UPDATE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--train_interval', type=int, default=TRAIN_INTERVAL)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--evaluation_episodes', type=int, default=EVALUATION_EPISODES)
    args = parser.parse_args()

    disable_eager_execution()

    actions = construct_action_space(
        n_steering=20,
        n_gas=10,
        n_brake=10,
    )

    n_actions = len(actions)

    model = construct_model(args.window_length, n_actions)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=1000000
    )

    memory = SequentialMemory(limit=args.memory_limit, window_length=args.window_length)
    processor = RaceCarEnvProcessor(actions=actions)

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=args.warmup_steps,
        gamma=.99,
        target_model_update=args.target_model_update,
        train_interval=args.train_interval,
        delta_clip=1.
    )

    dqn.compile(Adam(lr=args.learning_rate), metrics=['mae'])

    if args.mode == 'train':

        # TODO add tensorboard callbacks
        weights_filename = f'dqn_{env_name}_weights.h5f'
        checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env, callbacks=callbacks, nb_steps=args.steps, visualize=False, verbose=2)
        dqn.save_weights(weights_filename, overwrite=True)

        dqn.test(env, nb_episodes=args.evaluation_episodes, visualize=False)
        env.close()

    elif args.mode == 'test':
        weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=args.evaluation_episodes, visualize=True)
        env.close()
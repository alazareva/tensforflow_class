import argparse

import gym
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from gym.wrappers import GrayScaleObservation
from rl.processors import Processor
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.layers import Dense, Activation, Flatten, \
    Convolution2D, Permute, Input, Concatenate, Lambda, Add, Cropping2D, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.initializers import GlorotNormal
from gym import ActionWrapper, ObservationWrapper
from gym.spaces import Discrete, Box

INPUT_SHAPE = (96, 96)
WINDOW_LENGTH = 10
MEMORY_LIMIT = 1000000
WARMUP_STEPS = 20000
TARGET_MODEL_UPDATE = 10000
LEARNING_RATE = .00025
TRAIN_INTERVAL = 4
STEPS = 1750000
EVALUATION_EPISODES = 5

OTHER_INDICATORS = 8

# Determines if the car is in the grass (specific to CarRacing)
# Parameters:
# - state: A (nxmx1) image representing the current frame of the game
# Outputs:
# A boolean stating whether the car is in grass or not.
# Said to be in grass if  12x10 region around car has more than 44
# green pixels
def off_track(state):
    cropped = state[66:78, 43:53]
    green = np.sum(cropped[..., 1] >= 204)
    return green >= 45


def get_bottom_bar(img):
    is_off_track = int(off_track(img))
    h, w, _ = img.shape
    s = int(w / 40.0)
    h = h / 40.0
    bottom = grayscale_img(img[84:, :])
    threshold = 20
    black_and_white =  (bottom > threshold).astype('uint8') * 255
    x_start = 5 * s
    x_end = (x_start + 1) * s
    speed = black_and_white[:, x_start: x_end].mean()

    x_start = 7 * s
    x_end = (x_start + 1) * s
    wheel_0 = black_and_white[:, x_start: x_end].mean()

    x_start = 8 * s
    x_end = (x_start + 1) * s
    wheel_1 = black_and_white[:, x_start: x_end].mean()

    x_start = 9 * s
    x_end = (x_start + 1) * s
    wheel_2 = black_and_white[:, x_start: x_end].mean()

    x_start = 10 * s
    x_end = (x_start + 1) * s
    wheel_3 = black_and_white[:, x_start: x_end].mean()

    angle_mult = 0.8
    x_start = int((20 - 5 * angle_mult)) * s
    x_end = int((20 + 5 * angle_mult)) * s
    angle = black_and_white[:, x_start: x_end].mean()

    velocity_mul = 10
    x_start = int((30 - 0.4 * velocity_mul)) * s
    x_end = int((30 + 0.4 * velocity_mul)) * s
    velocity = black_and_white[:, x_start: x_end].mean()

    return np.array([speed, wheel_0, wheel_1, wheel_2, wheel_3, angle, velocity]) # is_off_track


class CarActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(CarActionWrapper, self).__init__(env)
        self._actions = [
            [0.0, 0.0, 0.0],  # Nothing
            [-1.0, 0.0, 0.0],  # Left
            [1.0, 0.0, 0.0],  # Right
            [0.0, 1.0, 0.0],  # Accelerate
            [0.0, 0.0, 0.8],  # break
        ]
        self.action_space = Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]

    def reverse_action(self, action):
        pass


class CarObservationWrapper(ObservationWrapper):
    def __index__(self, env):
        super(CarObservationWrapper, self).__init__(env)
        h, w = INPUT_SHAPE
        self.observation_space = Box(low=0, high=255, shape=(h + 1, w), dtype=np.uint8)

    def observation(self, observation):
        '''
        :param observation: numpy array of shape (96, 96, 3) return by the environment
        :return: 96 x 96 grayscale image
        '''
        rgb_weights = [0.2989, 0.5870, 0.1140]
        indicators = get_bottom_bar(observation)
        assert len(indicators) == OTHER_INDICATORS, f'Number of other indicators must be {OTHER_INDICATORS}'

        image = grayscale_img(observation)
        threshold = 150
        image = (image > threshold).astype('uint8') * 255
        padding = np.zeros(image.shape[1] - indicators.shape[0])
        other_info = np.concatenate([padding, indicators])  #  np.concatenate([indicators, padding])
        return np.vstack([image, other_info]) #observation.dot(rgb_weights)


def grayscale_img(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

# Mirrors the provided image vertically while preserving some parameters
# Parameters:
# - image: A (nxmx1) array of floats representing a grayscale image. Assumes
#          the provided image is a snapshot from the CarRacing game.
# Outputs:
# - A (nxmx1) array of floars representing a mirrored version of 'image'.
#   Reflects the gameplay screen (road + car + grass) vertically.
#   Reflects gyro and steering bars vertically
#   Preserves score, true speed, and ABS

def unit_image(image):
    image = np.array(image)
    return np.true_divide(image, 255)
# Processes image output by environment
# Parameters:
# - rgb_image: The RGB image output by the environment. A (nxmx3) array of floats
# - flip: A boolean. True if we wish to flip/reflect images
# - detect_edges: A boolean. True if we wish to apply canny edge detection to
#                 the images
# Outputs:
# An (nxmx1) array of floats representing the processed image
import cv2 as cv

def process_image(rgb_image):
    gray = grayscale_img(rgb_image)
    gray = unit_image(gray)
    return gray

class RaceCarEnvProcessor(Processor):

    def __init__(self, actions):
        self.actions = actions

    def process_observation(self, observation: np.ndarray):
        '''
        :param observation: numpy array of shape (96, 96, 3) return by the environment
        :return: 96 x 96 grayscale image
        '''
        rgb_weights = [0.2989, 0.5870, 0.1140]
        image = process_image(observation)
        other_info = np.full(image.shape[1], int(off_track(observation)))
        return np.vstack([image, other_info]) #observation.dot(rgb_weights)

    def process_action(self, action):
        return self.actions[action]

    def process_reward(self, reward):
        return np.clip(reward, a_min=-np.inf, a_max=1.0)

def construct_action_space(n_steering, n_gas, n_brake):
    steering_actions = np.zeros((n_steering, 3))
    steering_actions[:, 0] = np.linspace(-0.8, 0.8, n_steering)
    #gas_actions = np.zeros((n_gas, 3))
    #gas_actions[:, 1] = np.linspace(5, 1.0, n_gas)
    #brake_actions = np.zeros((n_brake, 3))
    #brake_actions[:, 2] = np.linspace(5, 1.0, n_brake)
    gas_actions = np.array([[0, 1, 0]])
    brake_actions = np.array([[0, 0, 1]])

    space = [
        [0, 0, 0.0], # Nothing
        [-1, 0, 0.0], # Left
        [+1, 0, 0.0], # Right
        [0, +1, 0.0], # Accelerat
        [0, 0, 0.8], # break
    ]

    return np.array(space) #np.vstack([steering_actions, gas_actions, brake_actions])


def construct_model(window_length, n_actions):
    model = Sequential()
    # (width, height, channels)
    input_shape = (window_length,) + INPUT_SHAPE
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer=GlorotNormal()))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer=GlorotNormal()))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer=GlorotNormal()))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=GlorotNormal()))
    model.add(Activation('relu'))
    model.add(Dense(n_actions, kernel_initializer=GlorotNormal()))
    model.add(Activation('linear'))
    print(model.summary())
    model.__setattr__('uses_learning_phase', False)
    return model

def construct_bi_model(window_length, n_actions):
    h, w = INPUT_SHAPE
    obs_shape = (window_length, h + 1, w)

    observation_input = Input(shape=obs_shape, name='observation_input')
    permute = Permute((2, 3, 1))(observation_input)

    image_slice = Cropping2D(cropping=((0, 1), (0, 0)))(permute)
    other_slice = Cropping2D(cropping=((h, 0), (0, 0)))(permute)
    print(image_slice)
    print(other_slice)

    image_slice = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Flatten()(image_slice)
    image_slice = Dense(512, activation='relu', kernel_initializer=GlorotNormal())(image_slice)

    other_slice = Dense(256, activation='relu', kernel_initializer=GlorotNormal())(other_slice)
    other_slice = Flatten()(other_slice)

    final = Concatenate()([image_slice, other_slice])

    out = Dense(n_actions, activation='linear', kernel_initializer=GlorotNormal())(final)
    model = Model(inputs=observation_input, outputs=out)
    print(model.summary())
    return model


def construct_bi_model_simple(window_length, n_actions):
    h, w = INPUT_SHAPE
    obs_shape = (window_length, h + 1, w)

    observation_input = Input(shape=obs_shape, name='observation_input')
    permute = Permute((2, 3, 1))(observation_input)

    image_slice = Cropping2D(cropping=((0, 1), (0, 0)))(permute)
    other_slice = Cropping2D(cropping=((h, 0), (0, 0)))(permute) # Cropping2D(cropping=((h, 0), (0, w - OTHER_INDICATORS)))(permute)

    image_slice = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Convolution2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=GlorotNormal())(image_slice)
    image_slice = Flatten()(image_slice)
    image_slice = Dense(512, activation='relu', kernel_initializer=GlorotNormal())(image_slice)

    other_slice = Dense(64, activation='relu', kernel_initializer=GlorotNormal())(other_slice)
    other_slice = Dense(32, activation='relu', kernel_initializer=GlorotNormal())(other_slice)
    other_slice = Flatten()(other_slice)

    conact = Concatenate()([image_slice, other_slice])

    out = Dense(256, activation='relu', kernel_initializer=GlorotNormal())(conact)

    out = Dense(n_actions, activation='linear', kernel_initializer=GlorotNormal())(out)
    model = Model(inputs=observation_input, outputs=out)
    print(model.summary())
    return model


def construct_critic(window_length, n_actions):
    action_input = Input(shape=(n_actions,), name='action_input')
    input_shape = (window_length,) + INPUT_SHAPE
    observation_input = Input(shape=input_shape, name='observation_input')
    p = Permute((2, 3, 1))(observation_input)
    flattened_observation = Flatten()(p)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic.__setattr__('uses_learning_phase', False)
    print(critic.summary())
    return critic, action_input


# python utils/karas_model.py --steps=100
# python utils/karas_model.py --mode=test --evaluation_episodes=2

if __name__=="__main__":

    env_name = 'CarRacing-v0'
    env = CarActionWrapper(CarObservationWrapper(gym.make(env_name)))

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
        n_steering=5,
        n_gas=2,
        n_brake=2,
    )

    n_actions = len(actions)
    print(actions)

    #model = construct_model(args.window_length, n_actions)
    #critic, action_input = construct_critic(args.window_length, n_actions)


    #model = construct_model(1, n_actions)
    #critic, action_input = construct_critic(1, n_actions)

    policy = EpsGreedyQPolicy(eps=0.05)
    '''LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=1000000
    )'''

    memory = SequentialMemory(limit=args.memory_limit, window_length=args.window_length)
    # processor = RaceCarEnvProcessor(actions=actions)

    model = construct_bi_model_simple(args.window_length, n_actions)

    agent = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        # processor=processor,
        nb_steps_warmup=args.warmup_steps,
        gamma=.99,
        target_model_update=args.target_model_update,
        train_interval=args.train_interval,
        delta_clip=1.,
        enable_dueling_network=True)
    agent.compile(Adam(lr=args.learning_rate), metrics=['mae'])

    if args.mode == 'train':

        # TODO add tensorboard callbacks
        weights_filename = f'dqn_{env_name}_weights.h5f'
        checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        agent.fit(env, callbacks=callbacks, nb_steps=args.steps, visualize=False, verbose=2)
        agent.save_weights(weights_filename, overwrite=True)

        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=False)
        env.close()

    elif args.mode == 'test':
        weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
        agent.load_weights(weights_filename)
        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=True)
        env.close()
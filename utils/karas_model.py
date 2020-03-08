import argparse
import logging
import gym
import numpy as np
from rl.agents.dqn import DQNAgent

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
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

MODEL_DIR = 'default'

MODEL_V2_NAME = 'simple_bi_v2'
MODEL_V1_NAME = 'simple_bi_v1'


def get_off_track(state):
    cropped = state[66:78, 43:53]
    green = np.sum(cropped[..., 1] >= 204)
    return int(green >= 45)


def get_bottom_bar_indicators(img):
    h, w, _ = img.shape
    s = int(w / 40.0)

    bottom = grayscale_img(img[84:, :])
    threshold = 20
    black_and_white = (bottom > threshold).astype('uint8') * 255
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

    return np.array([speed, wheel_0, wheel_1, wheel_2, wheel_3, angle, velocity])


class CarActionWrapper(ActionWrapper):
    ACTIONS = [
            [0.0, 0.0, 0.0],  # Nothing
            [-1.0, 0.0, 0.0],  # Left
            [1.0, 0.0, 0.0],  # Right
            [0.0, 1.0, 0.0],  # Accelerate
            [0.0, 0.0, 0.8],  # break
        ]
    def __init__(self, env):
        super(CarActionWrapper, self).__init__(env)
        self.action_space = Discrete(len(self.ACTIONS))

    def action(self, action):
        return self.ACTIONS[action]

    def reverse_action(self, action):
        pass

def grayscale_img(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

class CarObservationWrapperV2(ObservationWrapper):
    OTHER_INDICATORS = 8

    def __index__(self, env):
        super(CarObservationWrapperV2, self).__init__(env)
        h, w = INPUT_SHAPE
        self.observation_space = Box(low=0, high=255, shape=(h + 1, w),
                                     dtype=np.uint8)

    def observation(self, observation):
        indicators = get_bottom_bar_indicators(observation)
        is_off_track = get_off_track(observation)
        indicators = np.insert(indicators, 0, is_off_track)
        assert len(
            indicators) == self.OTHER_INDICATORS, f'Number of other indicators must be {self.OTHER_INDICATORS}'

        image = grayscale_img(observation)
        threshold = 150
        image = (image > threshold).astype('uint8') * 255
        padding = np.zeros(image.shape[1] - indicators.shape[0])
        other_info = np.concatenate([indicators, padding])
        return np.vstack([image, other_info])


class CarObservationWrapperV1(ObservationWrapper):
    OTHER_INDICATORS = 7

    def __index__(self, env):
        super(CarObservationWrapperV1, self).__init__(env)
        h, w = INPUT_SHAPE
        self.observation_space = Box(low=0, high=255, shape=(h + 1, w),
                                     dtype=np.uint8)

    def observation(self, observation):
        indicators = get_bottom_bar_indicators(observation)
        assert len(
            indicators) == self.OTHER_INDICATORS, f'Number of other indicators must be {self.OTHER_INDICATORS}'

        image = grayscale_img(observation)
        threshold = 150
        image = (image > threshold).astype('uint8') * 255
        padding = np.zeros(image.shape[1] - indicators.shape[0])
        other_info = np.concatenate([padding, indicators])
        return np.vstack([image, other_info])

class SimpleBiModelV2:
    OTHER_INDICATORS = 8

    def get_model(self, window_length, n_actions):
        h, w = INPUT_SHAPE
        obs_shape = (window_length, h + 1, w)

        observation_input = Input(shape=obs_shape, name='observation_input')
        permute = Permute((2, 3, 1))(observation_input)

        image_slice = Cropping2D(
            cropping=((0, 1), (0, 0)),
            name='crop_image'
        )(permute)
        other_slice = Cropping2D(
            cropping=((h, 0), (0, w - self.OTHER_INDICATORS)),
            name='crop_other_indicators',
        )(permute)

        image_slice = Convolution2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_1',
        )(image_slice)

        image_slice = Convolution2D(
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_2',
        )(image_slice)

        image_slice = Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_3',
        )(image_slice)

        image_slice = Flatten(name='flatten_image')(image_slice)
        image_slice = Dense(
            units=512,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='image_dense_1',
        )(image_slice)

        other_slice = Dense(
            units=128,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='indicators_dense_1',
        )(other_slice)
        other_slice = Dense(
            units=64,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='indicators_dense_2',
        )(other_slice)

        other_slice = Flatten(name='flatten_indicators')(other_slice)

        conact = Concatenate(name='concat_flat')([image_slice, other_slice])

        out = Dense(
            units=256,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='dense_all'
        )(conact)

        out = Dense(
            units=n_actions,
            activation='linear',
            kernel_initializer=GlorotNormal(),
            name='output',
        )(out)
        model = Model(inputs=observation_input, outputs=out, name=MODEL_V2_NAME)
        print(model.summary())
        return model

class SimpleBiModelV1:
    OTHER_INDICATORS = 7

    def get_model(self, window_length, n_actions):
        h, w = INPUT_SHAPE
        obs_shape = (window_length, h + 1, w)

        observation_input = Input(
            shape=obs_shape,
            name='observation_input'
        )
        permute = Permute(dims=(2, 3, 1), name='permute_dims')(observation_input)

        image_slice = Cropping2D(cropping=((0, 1), (0, 0)), name='crop_image')(permute)
        other_slice = Cropping2D(cropping=((h, 0), (0, 0)), name='crop_indicators')(permute)

        image_slice = Convolution2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_1',
        )(image_slice)
        image_slice = Convolution2D(
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_2'
        )(image_slice)
        image_slice = Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='conv_3',
        )(image_slice)

        image_slice = Flatten(name='flatten_image')(image_slice)
        image_slice = Dense(
            units=512,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='dense_image'
        )(image_slice)

        other_slice = Dense(
            units=128,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='dense_indicators_1',
        )(other_slice)
        other_slice = Dense(
            units=64,
            activation='relu',
            kernel_initializer=GlorotNormal(),
            name='dense_indicators_2'
        )(other_slice)
        other_slice = Flatten(name='flatten_indicators')(other_slice)

        conact = Concatenate('concat_all')([image_slice, other_slice])

        out = Dense(
            units=n_actions,
            activation='linear',
            kernel_initializer=GlorotNormal(),
            name='output'
        )(conact)
        model = Model(inputs=observation_input, outputs=out, name=MODEL_V1_NAME)
        print(model.summary())
        return model


def get_observation_wrapper(model_name):
    if model_name == MODEL_V2_NAME:
        return CarObservationWrapperV2
    if model_name == MODEL_V1_NAME:
        return CarObservationWrapperV1

def get_model_builder(model_name):
    if model_name == MODEL_V2_NAME:
        return SimpleBiModelV2
    if model_name == MODEL_V1_NAME:
        return SimpleBiModelV1

# python utils/karas_model.py --steps=100
# python utils/karas_model.py --mode=test --evaluation_episodes=2

if __name__=="__main__":

    disable_eager_execution()

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
    parser.add_argument('--model', type=str, default=MODEL_V2_NAME)
    parser.add_argument('--load_weights_from', type=str, default=None)
    args = parser.parse_args()

    model_name = args.model
    print(f"Selected Model: {model_name}")
    observation_wrapper = get_observation_wrapper(model_name)
    model_builder = get_model_builder(model_name)

    env_name = 'CarRacing-v0'
    env = CarActionWrapper(observation_wrapper(gym.make(env_name)))

    n_actions = len(CarActionWrapper.ACTIONS)

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

    model = model_builder().get_model(args.window_length, n_actions)

    agent = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=args.warmup_steps,
        gamma=.99,
        target_model_update=args.target_model_update,
        train_interval=args.train_interval,
        delta_clip=1.,
        enable_dueling_network=True)

    agent.compile(Adam(lr=args.learning_rate), metrics=['mae'])

    if args.load_weights_from is not None:
        print(f"Loading Weights From: {args.load_weights_from}")
        weights_filename = f'{args.load_weights_from}/' + 'dqn_{}_weights.h5f'.format(env_name)
        agent.load_weights(weights_filename)

    if args.mode == 'train':
        import os
        current_directory = os.getcwd()
        model_weight_dir = os.path.join(current_directory, model_name)
        if not os.path.exists(model_weight_dir):
            os.makedirs(model_weight_dir)

        # TODO add tensorboard callbacks
        weights_filename = f'{model_name}/dqn_{env_name}_weights.h5f'
        checkpoint_weights_filename = f'{model_name}/dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = f'{model_name}/' + 'dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        agent.fit(env, callbacks=callbacks, nb_steps=args.steps, visualize=False, verbose=2)
        agent.save_weights(weights_filename, overwrite=True)

        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=False)
        env.close()

    elif args.mode == 'test':
        weight_dir = args.load_weights_from or model_name
        weights_filename = f'{weight_dir}/' + 'dqn_{}_weights.h5f'.format(env_name)
        agent.load_weights(weights_filename)
        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=True)
        env.close()
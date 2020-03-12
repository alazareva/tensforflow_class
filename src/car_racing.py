import os
import argparse
from datetime import datetime
import warnings

import gym
from gym import ActionWrapper, ObservationWrapper, RewardWrapper
from gym.spaces import Discrete, Box
from gym.wrappers import Monitor
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Flatten, \
    Convolution2D, Permute, Input, Cropping2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import summary_ops_v2

INPUT_SHAPE = (96, 96)
WINDOW_LENGTH = 10
MEMORY_LIMIT = 1000000
WARMUP_STEPS = 20000
TARGET_MODEL_UPDATE = 10000
LEARNING_RATE = .00025
TRAIN_INTERVAL = 4
STEPS = 1750000
EVALUATION_EPISODES = 5

MODEL_NAME = 'simple_bi_model'

def grayscale_img(image):
    """
    Converts a color image to grayscale numpy array
    """
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def get_bottom_bar_indicators(img):
    """
    Converts visual bar graphs of true speed, four ABS sensors, steering wheel position and velocity 
    at the bottom of screen to average values in a numpy array. 

    """
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
    """
    Wrapper with possible actions the car can make at each step.
    Do noting, turn left, turn right, accelerate and brake.
    """
    ACTIONS = [
            [0.0, 0.0, 0.0],  # Nothing
            [-1.0, 0.0, 0.0],  # Left
            [1.0, 0.0, 0.0],  # Right
            [0.0, 1.0, 0.0],  # Accelerate
            [0.0, 0.0, 0.8],  # Brake
        ]
    def __init__(self, env):
        super(CarActionWrapper, self).__init__(env)
        self.action_space = Discrete(len(self.ACTIONS))

    def action(self, action):
        return self.ACTIONS[action]

class CarObservationWrapper(ObservationWrapper):
    OTHER_INDICATORS = 7

    def __index__(self, env):
        super(CarObservationWrapper, self).__init__(env)
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

class CarRewardWrapper(RewardWrapper):

    def reward(self, reward):
        return min(reward, 1)

class SimpleBiModel:
    """
    Building the model:
        Output image is cropped into a track image and a indicators image.
        The track image is moved through three convolutional layers, then flattened, before being put though a dense layer.
        The indicator image is put through three dense layers, then flattened before being concatenated with the track image.
        
    """
    @staticmethod
    def get_model(window_length, n_actions):
        h, w = INPUT_SHAPE
        obs_shape = (window_length, h + 1, w)

        observation_input = Input(
            shape=obs_shape,
            name='observation_input'
        )
        permute = Permute(dims=(2, 3, 1), name='permute_dims')(observation_input)

        image_slice = Cropping2D(cropping=((0, 1), (0, 0)), name='crop_image')(permute) # track image
        other_slice = Cropping2D(cropping=((h, 0), (0, 0)), name='crop_indicators')(permute) # indicators at bottom of screen
        
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

        conact = Concatenate(name='concat_all')([image_slice, other_slice])

        out = Dense(
            units=n_actions,
            activation='linear',
            kernel_initializer=GlorotNormal(),
            name='output'
        )(conact)
        model = Model(inputs=observation_input, outputs=out, name=MODEL_NAME)
        print(model.summary())
        return model

class TensorboardCallback(Callback):
    """
    Logging metrics for Tensorboard
    """
    def __init__(self, log_dir='tf_logs', mode='train'):
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        logs = f'{log_dir}/{mode}'
        self.mode = mode
        self.writer = tf.summary.create_file_writer(logdir=logs)

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        self.writer.close()

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """Report progress to Tensorboard"""
        if self.mode == 'train':
            metrics = np.array(self.metrics[episode])
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                for idx, name in enumerate(self.metrics_names):
                    try:
                        value = np.nanmean(metrics[:, idx])
                        self._log_scalars([(name, value)], episode=episode + 1)
                    except Warning:
                        pass
        self._log_scalars(
            logs=[('episode_reward', np.sum(self.rewards[episode])),
                  ('reward_mean', np.mean(self.rewards[episode]))
                  ],
            episode=episode+1,
        )

        # Free up resources.
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def _log_scalars(self, logs, episode):
        with summary_ops_v2.always_record_summaries():
            with self.writer.as_default():
                for (name, value) in logs:
                    tf.summary.scalar(name, value, step=episode)

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        if self.mode == 'train':
            self.metrics[episode].append(logs['metrics'])
        self.step += 1


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'record'], default='train')
    parser.add_argument('--window_length', type=int, default=WINDOW_LENGTH)
    parser.add_argument('--memory_limit', type=int, default=MEMORY_LIMIT)
    parser.add_argument('--warmup_steps', type=int, default=WARMUP_STEPS)
    parser.add_argument('--target_model_update', type=int, default=TARGET_MODEL_UPDATE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--train_interval', type=int, default=TRAIN_INTERVAL)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--evaluation_episodes', type=int, default=EVALUATION_EPISODES)
    parser.add_argument('--load_weights_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=MODEL_NAME)
    args = parser.parse_args()

    env_name = 'CarRacing-v0'
    env = CarActionWrapper(CarObservationWrapper(gym.make(env_name)))
    if args.mode == 'train':
        env = CarRewardWrapper(env)

    n_actions = len(CarActionWrapper.ACTIONS)

    policy = EpsGreedyQPolicy(eps=0.05)

    memory = SequentialMemory(limit=args.memory_limit, window_length=args.window_length)

    model = SimpleBiModel().get_model(args.window_length, n_actions)

    tb_log_dir = 'tensorboard'
    tb_logs = f'{tb_log_dir}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    graph_dir = f'{tb_logs}/graph'
    writer = tf.summary.create_file_writer(logdir=graph_dir)
    # save the graph
    with writer.as_default():
        summary_ops_v2.graph(K.get_graph(), step=0)
    writer.close()

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

    if args.load_weights_from is not None and args.mode != 'record':
        print(f"Loading Weights From: {args.load_weights_from}")
        weights_filename = f'{args.load_weights_from}/' + 'dqn_{}_weights.h5f'.format(env_name)
        agent.load_weights(weights_filename)

    if args.mode == 'train':
        import os
        current_directory = os.getcwd()
        model_weight_dir = os.path.join(current_directory, args.save_dir)
        if not os.path.exists(model_weight_dir):
            os.makedirs(model_weight_dir)

        weights_filename = f'{args.save_dir}/dqn_{env_name}_weights.h5f'
        checkpoint_weights_filename = f'{args.save_dir}/dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = f'{args.save_dir}/' + 'dqn_{}_log.json'.format(env_name)
        callbacks = [
            ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000),
            FileLogger(log_filename, interval=100),
            TensorboardCallback(log_dir=tb_logs)
        ]
        agent.fit(env, callbacks=callbacks, nb_steps=args.steps, visualize=False, verbose=2)
        agent.save_weights(weights_filename, overwrite=True)

        callbacks = [TensorboardCallback(log_dir=tb_logs, mode='test')]
        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=False, callbacks=callbacks)
        env.close()

    elif args.mode == 'test':
        callbacks = [TensorboardCallback(log_dir=tb_logs, mode='test')]
        agent.test(env, nb_episodes=args.evaluation_episodes, visualize=True, callbacks=callbacks)
        env.close()

    elif args.mode == 'record':
        weight_dir = args.load_weights_from
        for sub_dir in os.listdir(weight_dir):
            if sub_dir != '.DS_Store':
                weights_filename = f'{weight_dir}/' + f'{sub_dir}/' + 'dqn_{}_weights.h5f'.format(env_name)
                print(weights_filename)
                agent.load_weights(weights_filename)
                callbacks = [TensorboardCallback(log_dir=tb_logs, mode='test')]
                env = Monitor(env, 'monitor_files_'+sub_dir, force=True, uid=True,
                              video_callable=lambda episode_id: True, write_upon_reset=True)
                agent.test(env, nb_episodes=args.evaluation_episodes, visualize=False, callbacks=callbacks)
                env.close()


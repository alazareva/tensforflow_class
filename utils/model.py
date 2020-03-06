

class Agent:
    def save(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, **kwargs):
        raise NotImplementedError

    def initialize(self, **kwargs):
        raise NotImplementedError

    def start(self, state):
        raise NotImplementedError

    def step(self, reward, state):
        raise NotImplementedError

    def end(self, reward):
        raise NotImplementedError

class DeepQAgent(Agent):
    pass

class SarsaAgent(Agent):
    pass



def train():
    pass

def play():
    pass
    """
    read model file
    for each step
    use model to predict value and select next step
    render next step
    """

import gym
from  tensorflow.keras import Model
import numpy as np
class Agent:
    def __init__(self, model: Model):
        self.model = model
        self.nb_actions = 10
        self.batch_size = 100
        self.training = True
        self.epsilon = 0.05
        self.gamma = 0.2

        self.reset_states()

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()

    env_name = 'CarRacing-v0'
    env = gym.make(env_name)

   # policy = None
    #def reset_states(self):
     #   pass
    # TODO memory = SequentialMemory https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py

    def greedy(self, q_values):
        return np.argmax(q_values)

    def epsilon_greedy(self, q_values):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.nb_actions)
        return self.greedy(q_values)

    def forward(self, observation):
        # Select an action.
        # TODO here state is a sequence off observations we can assume it's of lenght 1?
        state = self.memory.get_recent_state(observation) # TODO figure out if this can just use recent
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.epsilon_greedy(q_values=q_values)
        else:
            action = self.greedy(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        if self.training:
            if self.step != 0 and self.step % self.train_interval == 0:
                experiences = self.memory.sample(self.batch_size)

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)

                # Prepare and validate parameters.
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)

                # Compute Q values for mini-batch update.
                target_q_values = self.model.predict_on_batch(state1_batch)
                q_batch = np.max(target_q_values, axis=1).flatten()

                targets = np.zeros((self.batch_size, self.nb_actions))
                dummy_targets = np.zeros((self.batch_size,))
                masks = np.zeros((self.batch_size, self.nb_actions))

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * q_batch
                # Set discounted reward to zero for all states that were terminal.
                discounted_reward_batch *= terminal1_batch
                Rs = reward_batch + discounted_reward_batch
                for idx, (target, mask, R, action) in enumerate(
                    zip(targets, masks, Rs, action_batch)):
                    target[action] = R  # update action with estimated accumulated reward
                    dummy_targets[idx] = R
                    mask[action] = 1.  # enable loss for this specific action
                targets = np.array(targets).astype('float32')
                masks = np.array(masks).astype('float32')

                # Finally, perform a single update on the entire batch. We use a dummy target since
                # the actual loss is computed in a Lambda layer that needs more complex input. However,
                # it is still useful to know the actual target to compute metrics properly.
                ins = [state0_batch] if type(
                    self.model.input) is not list else state0_batch
                metrics = self.trainable_model.train_on_batch(
                    ins + [targets, masks], [dummy_targets, targets])
                metrics = [metric for idx, metric in enumerate(metrics) if
                           idx not in (1, 2)]  # throw away individual losses
                metrics += self.policy.metrics
                if self.processor is not None:
                    metrics += self.processor.metrics

            if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
                self.update_target_model_hard()

            return metrics


    def compute_q_values(self, state):
        q_values = self.model.predict_on_batch([state])
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.combined_model.reset_states()
            self.target_V_model.reset_states()

    def fit(self):
        processor = None # TODO get this thing
        observation = None
        episode = 0
        step = 0
        max_total_steps = 100000
        episode_step = None
        episode_reward = None
        done = False
        nb_max_episode_steps = 1000
        while step < max_total_steps:
            if observation is None: # start new espisode
                observation = self.env.reset()
                observation = processor.process_observation(observation)
                episode_step = 0
                episode_reward = 0
            # process step
            action = self.forward(observation)
            action = self.processor.process_action(action)
            observation, reward, done, info = self.env.step(action)
            observation = processor.process_observation(observation)
            metrics = self.backward(reward, terminal=done)
            episode_reward += reward
            episode_step += 1
            step += 1

            if done:
                # episode ended
                self.forward(observation)
                self.backward(0., terminal=False)

                episode += 1
                observation = None
                episode_step = None
                episode_reward = None

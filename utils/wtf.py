from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo.examples.v1 import train_eval_random_py_env
from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec

FLAGS = flags.FLAGS


def env_load_fn(env_name):
  del env_name
  obs_spec = array_spec.BoundedArraySpec((2,), np.int32, -10, 10)
  action_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
  return random_py_environment.RandomPyEnvironment(
      obs_spec, action_spec=action_spec, min_duration=2, max_duration=4)



if __name__ == '__main__':
  app.run(train_eval_random_py_env.main)
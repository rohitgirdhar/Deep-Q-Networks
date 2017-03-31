#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import copy
import random
import gym
# import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, RMSprop

import deeprl as tfrl
from deeprl import dqn
from deeprl.objectives import mean_huber_loss

from deeprl import networks
from deeprl import preprocessors
from deeprl import policy
from deeprl import memory
from deeprl import objectives

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    pass


def get_output_folder(parent_dir, env_name, exp_id=None):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    try:
      # os.makedirs(parent_dir, exist_ok=True)
      os.makedirs(parent_dir)
    except:
      pass
    if exp_id is None:
      experiment_id = 0
      for folder_name in os.listdir(parent_dir):
          if not os.path.isdir(os.path.join(parent_dir, folder_name)):
              continue
          try:
              # folder_name = int(folder_name.split('-run')[-1])
              folder_name = int(folder_name.split('_')[0])
              if folder_name > experiment_id:
                  experiment_id = folder_name
          except:
              pass
      experiment_id += 1
    else:
      experiment_id = exp_id

    parent_dir = os.path.join(parent_dir,
                              '{0:03d}_'.format(experiment_id) + env_name)
    return parent_dir, experiment_id


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', default='0', type=str, help='GPU to use.')
    parser.add_argument('--optimizer', default='rmsprop', type=str,
                        help='optimizer (rmsprop/adam).')
    parser.add_argument('--learning_rate', default=0.00025, type=float,
                        help='Learning rate.')
    parser.add_argument('--model', default='convnet', type=str,
                        help='Type of model to use.')
    parser.add_argument('--max_iters', default=100000000, type=int,
                        help='Max num of iterations to run for.')
    parser.add_argument('--checkpoint',
                        default='',
                        type=str,
                        help='Checkpoint to load from.')
    parser.add_argument('--render', action='store_true',
                        default=False,
                        help='Render what we got, or train?')
    parser.add_argument('--render_path', type=str,
                        default='/dev/null/',
                        help='Path to store the render in.')
    parser.add_argument('--std_img', action='store_true',
                        default=False,
                        help='Standardize (-1,1) the image or not.')
    parser.add_argument('--part_gpu', action='store_true',
                        default=True,
                        help='Use part of GPU.')
    parser.add_argument('--train_policy', type=str,
                        default='anneal',
                        help='anneal/epgreedy')
    parser.add_argument('--exp_id', type=int,
                        default=None,
                        help='For assoc between scripts and results, '
                             'give script number.')
    parser.add_argument('--target_update_freq', type=int,
                        default=10000,
                        help='Sync the target and live networks.')
    parser.add_argument('--train_freq', type=int,
                        default=4,
                        help='Number of iters to push into replay before training.')
    parser.add_argument('--mem_size', type=int,
                        default=100000,
                        help='Size of replay memory, 1M is too large.')
    parser.add_argument('--learning_type', type=str,
                        default='normal',
                        help='Set normal, or double for DDQN.')
    parser.add_argument('--final_eval', action='store_true',
                        default=False,
                        help='Perform the final 100 episode evaluation.')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    args.output, experiment_id = get_output_folder(args.output, args.env,
                                                   args.exp_id)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.part_gpu:
      from keras.backend.tensorflow_backend import set_session
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      set_session(tf.Session(config=config))

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    input_size = 84
    frame_history = 4
    batch_size = 32
    input_shape = (input_size, input_size, frame_history)
    # mem_size = 1000000
    mem_size = args.mem_size  # unable to initialize 1M memory, same as achal
    gamma = 0.99
    target_update_freq = args.target_update_freq
    num_burn_in = 20000
    train_freq = args.train_freq

    summary_writer = None
    if not args.render and not args.final_eval:
      summary_writer = tf.summary.FileWriter(logdir=args.output)
    env = gym.make(args.env)
    env.seed(args.seed)
    env.reset()

    model = networks.get_model(args.model, input_shape, env.action_space.n)
    target_model = networks.get_model(args.model, input_shape, env.action_space.n)
    preproc = preprocessors.PreprocessorSequence(
      [preprocessors.AtariPreprocessor(input_size, args.std_img),
       preprocessors.HistoryPreprocessor(frame_history)])
    if args.train_policy == 'anneal':
      pol = policy.LinearDecayGreedyEpsilonPolicy(
        1, 0.1, 1000000)
    elif args.train_policy == 'epgreedy':
      pol = policy.GreedyEpsilonPolicy(0.1)
    else:
      raise ValueError()

    if args.optimizer == 'rmsprop':
      optimizer = RMSprop(args.learning_rate, rho=0.95, epsilon=0.01)
    elif args.optimizer == 'adam':
      optimizer = Adam(args.learning_rate)
    mem = memory.BasicReplayMemory(mem_size)
    learning_type = dqn._LEARNING_TYPE_NORMAL
    if args.learning_type == 'double':
      learning_type = dqn._LEARNING_TYPE_DOUBLE
    D = dqn.DQNAgent(
        model,
        target_model,
        preproc,
        mem,
        pol,
        gamma,
        target_update_freq,
        num_burn_in,
        train_freq,
        batch_size,
        optimizer=optimizer,
        loss_func='mse',
        summary_writer=summary_writer,
        checkpoint_dir=args.output,
        experiment_id=experiment_id,
        env_name=args.env,
        learning_type=learning_type)

    if args.checkpoint:
      D.load(args.checkpoint)
    if args.final_eval:
      D.evaluate(env, 100, 0, max_episode_length=1e4, final_eval=True)
    elif args.render:
      # testing_env = copy.deepcopy(env)
      D.evaluate(env, 1, 0, max_episode_length=1e4,
                 render=True, render_path=args.render_path)
    else:
      D.fit(env, args.max_iters)

if __name__ == '__main__':
    main()

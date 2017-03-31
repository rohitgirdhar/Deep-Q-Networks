import numpy as np
import copy
import tensorflow as tf
import time
import sys
from StringIO import StringIO
import matplotlib.pyplot as plt
import gym
import os

from deeprl import preprocessors
from deeprl import policy
from keras.models import load_model

PRINT_AFTER_ITER = 100
EVAL_AFTER_ITER = 5e4
EVAL_MAX_EPISODE_LEN = 1e4
NUM_TEST_EPISODES = 20
SAVE_AFTER_ITER = 5e4
NO_OP_STEPS = 30  # for these many steps, do nothing in each episode

_LEARNING_TYPE_NORMAL = 0
_LEARNING_TYPE_DOUBLE = 1

"""Main DQN agent."""
class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 target_q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 optimizer,
                 loss_func,
                 summary_writer,
                 checkpoint_dir,
                 experiment_id,
                 env_name,
                 learning_type=_LEARNING_TYPE_NORMAL):  # normal/double
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.target_q_network.set_weights(
          self.q_network.get_weights())
        self.compile(optimizer, loss_func)
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.summary_writer = summary_writer
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id
        self.env_name = env_name
        self.training_reward_seen = 0
        # The following are only there to make it fast at runtime (avoid alloc)
        self.input_batch = np.zeros([batch_size,] + \
                                    list(q_network.input_shape[-3:]), dtype='float')
        self.nextstate_batch = np.zeros([batch_size,] + \
                                        list(q_network.input_shape[-3:]), dtype='float')
        self.learning_type = learning_type

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.

        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_network.compile(optimizer, loss_func)
        self.target_q_network.compile(optimizer, loss_func)

    def calc_q_values(self, state, preproc=None, network=None):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if preproc is None:
          preproc = self.preprocessor
        if network is None:
          network = self.q_network
        return self.q_network.predict(np.expand_dims(
          preproc.process_state_for_network(state), 0))

    def update_policy(self, itr):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        samples = self.memory.sample(self.batch_size)
        ns = len(samples)
        assert(ns == self.batch_size)
        # target_batch = np.zeros([ns, self.q_network.output_shape[-1]],
        #                         dtype='float')
        if self.learning_type == _LEARNING_TYPE_DOUBLE:
          # randomly swap the target and active networks
          if np.random.uniform() < 0.5:
            temp = self.q_network
            self.q_network = self.target_q_network
            self.target_q_network = temp
        for i in range(ns):
          state, _, _, nextstate, _ = samples[i]
          self.input_batch[i, ...] = state
          self.nextstate_batch[i, ...] = nextstate
        self.input_batch = self.preprocessor.process_state_for_network(
          self.input_batch)
        self.nextstate_batch = self.preprocessor.process_state_for_network(
          self.nextstate_batch)
        target_batch = self.q_network.predict(self.input_batch)
        nextstate_q_values = self.target_q_network.predict(self.nextstate_batch)
        if self.learning_type == _LEARNING_TYPE_DOUBLE:
          nextstate_q_values_live_network = self.q_network.predict(
            self.nextstate_batch)
        for i in range(ns):
          # to incur 0 loss on all actions but the one we care about,...
          # target_batch[i, ...] = cur_q_values[i, ...]
          _, action, reward, _, is_terminal = samples[i]
          if is_terminal:
            target_batch[i, action] = reward
          else:
            if self.learning_type == _LEARNING_TYPE_DOUBLE:
              selected_action = np.argmax(nextstate_q_values_live_network[i].flatten())
              target_batch[i, action] = reward + self.gamma * \
                  nextstate_q_values[i, selected_action]
            else:
              target_batch[i, action] = reward + self.gamma * np.max(
                nextstate_q_values[i])
        self.training_reward_seen += sum([el[2] for el in samples])
        if itr % PRINT_AFTER_ITER == 0:
          # add image summary
          im_summaries = []
          for k in range(3):
            s = StringIO()
            plt.imsave(s, np.mean(self.input_batch[k], axis=-1), format='png')
            img_sum = tf.Summary.Image(
              encoded_image_string=s.getvalue(),
              height=self.input_batch[k].shape[0],
              width=self.input_batch[k].shape[1])
            im_summaries.append(tf.Summary.Value(
              tag='input/{}'.format(k), image=img_sum))
          self.summary_writer.add_summary(tf.Summary(
            value=im_summaries),
            global_step=itr)
        loss = self.q_network.train_on_batch(self.input_batch, target_batch)
        return loss

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        # first fill up the replay memory
        self.preprocessor.reset()
        env_current_state = env.reset()
        # env_current_state = self.run_no_op_steps(env)
        env_current_state = \
          self.preprocessor.process_state_for_memory(env_current_state)
        # any env.reset must be followed by preprocessor reset (it's a
        # stateful function)
        testing_env = copy.deepcopy(env)
        value_fn = np.random.random((env.action_space.n,))
        for _ in range(self.num_burn_in):
          env_current_state = self.push_replay_memory(
            env_current_state, env,
            policy.UniformRandomPolicy(env.action_space.n),
            is_training=False, value_fn=value_fn)
        start_time = time.time()
        for itr in range(num_iterations):
          value_fn = self.calc_q_values(
            env_current_state, network=self.q_network)
          env_current_state = self.push_replay_memory(
            env_current_state, env, self.policy,
            is_training=True, value_fn=value_fn)

          if itr % self.target_update_freq == 0:
            self.target_q_network.set_weights(
              self.q_network.get_weights())

          if itr % self.train_freq == 0:
            loss = self.update_policy(itr)

          if itr % PRINT_AFTER_ITER == 0:
            print('Iteration {:}: Loss {:.12f} ({:.4f} it/sec) '
                  '(reward seen: {})'.format(
                  itr, loss, PRINT_AFTER_ITER * 1.0 / (time.time()-start_time),
                  self.training_reward_seen))
            start_time = time.time()
            self.summary_writer.add_summary(tf.Summary(value=[
              tf.Summary.Value(
                tag='loss',
                simple_value=loss.item())]),
              global_step=itr)

          if itr % EVAL_AFTER_ITER == 0:
            self.evaluate(testing_env, NUM_TEST_EPISODES, itr,
                          max_episode_length=EVAL_MAX_EPISODE_LEN)
          if itr % SAVE_AFTER_ITER == 0:
            self.save(itr)

    def push_replay_memory(
        self, env_current_state, env, policy,
        is_training, value_fn):
        dont_reset = False
        env_current_lives = -1
        try:
          env_current_lives = env.env.ale.lives()
        except:
          pass
        action = policy.select_action(q_values=value_fn, is_training=is_training)
        nextstate, reward, is_terminal, debug_info = env.step(action)
        if 'ale.lives' in debug_info:
          if debug_info['ale.lives'] < env_current_lives:
            if not is_terminal:
              dont_reset = True
            is_terminal = True
        # enduro seems to give (-) rewards on hitting the sides... (check?)
        reward = self.preprocessor.process_reward(reward)
        nextstate = self.preprocessor.process_state_for_memory(nextstate)
        self.memory.append(env_current_state, action,
                           reward, nextstate, is_terminal)
        if is_terminal and not dont_reset:
          self.preprocessor.reset()
          nextstate = env.reset()
          # nextstate = self.run_no_op_steps(env)
          nextstate = self.preprocessor.process_state_for_memory(nextstate)
        return nextstate

    def save(self, itr):
        filename = "%s/%s_run%d_iter%d.h5" % (self.checkpoint_dir, self.env_name, self.experiment_id, itr)
        self.q_network.save(filename)

    def load(self, filename):
        # self.q_network = load_model(filename)
        # The above does not work for models with lambda functions
        self.q_network.load_weights(filename)

    def evaluate(self, env, num_episodes, itr, max_episode_length=None,
                 render=False, render_path='', final_eval=False):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print('Running evaluation')
        preproc = preprocessors.PreprocessorSequence(
          [preprocessors.AtariPreprocessor(
            self.q_network.input_shape[1],
            self.preprocessor.preprocessors[0].std_img),
           preprocessors.HistoryPreprocessor(self.q_network.input_shape[-1])])
        pol = policy.GreedyEpsilonPolicy(0.05)
        # pol = policy.UniformRandomPolicy(env.action_space.n)
        all_stats = []
        all_rewards = []
        for ep_id in range(num_episodes):
          print('Running episode {}'.format(ep_id))
          if render:
            env = gym.wrappers.Monitor(
              env,
              render_path,
              force=True)
          nextstate = env.reset()
          # nextstate = self.run_no_op_steps(env)
          preproc.reset()
          is_terminal = False
          stats = {
            'total_reward': 0,
            'episode_length': 0,
            'max_q_value': 0,
          }
          while not is_terminal and \
            stats['episode_length'] < max_episode_length:
            nextstate = preproc.process_state_for_memory(nextstate)
            q_values = self.calc_q_values(nextstate, preproc)
            action = pol.select_action(q_values=q_values)
            nextstate, reward, is_terminal, _ = env.step(action)
            stats['total_reward'] += reward
            stats['episode_length'] += 1
            stats['max_q_value'] += max(q_values)
          all_stats.append(stats)
          all_rewards.append(stats['total_reward'])
          print('Current mean+std: {} {}'.format(np.mean(all_rewards),
                                                 np.std(all_rewards)))
        # aggregate the stats
        final_stats = {}
        if render:
          return  # no need to log this interaction
        if final_eval:
          print('Mean reward: {}'.format(np.mean(all_rewards)))
          print('Std reward: {}'.format(np.std(all_rewards)))
          return
        for key in all_stats[0]:
          final_stats['mean_' + key] = np.mean([el[key] for el in all_stats]).item()
          self.summary_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(
              tag='eval/{}'.format(key),
              simple_value=final_stats['mean_' + key])]),
            global_step=itr)
        print('Evaluation result: {}'.format(final_stats))

    def run_no_op_steps(self, env):
      for _ in range(NO_OP_STEPS-1):
        _, _, is_terminal, _ = env.step(0)
        if is_terminal:
          env.reset()
      nextstate, _, is_terminal, _ = env.step(0)
      if is_terminal:
        nextstate = env.reset()
      return nextstate

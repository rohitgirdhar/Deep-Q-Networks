import numpy as np
from PIL import Image

from deeprl.core import ReplayMemory
from deeprl.core import Sample


class BasicSample():
  def __init__(self):
    self.state = np.zeros((84, 84, 4), dtype=np.uint8)
    self.action = 0
    self.reward = 0.0
    self.nextstate = np.zeros((84, 84, 4), dtype=np.uint8)
    self.is_terminal = False

  def assign(self, state, action, reward, nextstate, is_terminal):
    self.state[:] = state
    self.action = action
    self.reward = reward
    self.nextstate[:] = nextstate
    self.is_terminal = is_terminal

class BasicReplayMemory():
  def __init__(self, max_size):
    # initialize the whole memory at once
    self.memory = [BasicSample() for _ in range(max_size)]
    self.max_size = max_size
    self.itr = 0  # insert the next element here
    self.cur_size = 0

  def append(self, state, action, reward, nextstate, is_terminal):
    self.memory[self.itr].assign(
      state, action, reward, nextstate, is_terminal)
    self.itr += 1
    self.cur_size = min(self.cur_size + 1, self.max_size)
    self.itr %= self.max_size

  def sample(self, batch_size):
    res = []
    for i, idx in enumerate(np.random.randint(0, self.cur_size, size=batch_size)):
      sample = self.memory[idx]
      res.append((sample.state,
                  sample.action,
                  sample.reward,
                  sample.nextstate,
                  sample.is_terminal))
    return res

  def get_size(self):
    return self.cur_size

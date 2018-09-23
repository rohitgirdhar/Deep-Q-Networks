"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
      self.history = np.zeros((84, 84, history_length))
      self.history_length = history_length

    def process_state_for_memory(self, state):
      """You only want history when you're deciding the current action to take."""
      # push this frame to history list
      # asserts take up time, so removing now
      # assert(np.all(state.shape == self.history[..., 0].shape))
      self.history[..., 0] = state
      self.history = np.roll(self.history, -1, axis=-1)
      return self.history.copy()

    def reset(self):
      """Reset the history sequence.
      Useful when you start a new episode.
      """
      self.history = np.zeros((84, 84, self.history_length))

    def get_config(self):
      return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size, std_img):
        self.new_size = new_size
        self.resize_size = (110, 84)
        self.std_img = std_img

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        I = Image.fromarray(state, 'RGB')
        I = I.convert('L')  # to gray
        I = I.resize((self.new_size, self.new_size), Image.ANTIALIAS)
        # following are not done in the nature paper, so ignoring all that.
        # I = I.resize(self.resize_size)
        # width, height = I.size   # Get dimensions
        # new_width = self.new_size
        # new_height = self.new_size
        # left = max((width - new_width)/2, 0)
        # top = max((height - new_height)/2, 0)
        # right = min((width + new_width)/2, width)
        # bottom = min((height + new_height)/2, height)
        # I = I.crop((left, top, right, bottom))
        I = np.array(I).astype('uint8')
        return I

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        # no need of any processing since frames stored in memory are already
        # resized/cropped etc
        # removing the following assert as I want all sizes of images to be
        # processed
        # assert(state.shape[:2] == (self.new_size, self.new_size))
        state = state.astype('float')
        # not done in nature paper, so not doing this
        if self.std_img:
          state -= 128.0
          state /= 255.0
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return [self.process_state_for_network(sample) for
                sample in samples]

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.sign(reward)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process_state_for_network(self, state):
        """
        simply runs all the pre-processors
        """
        for preproc in self.preprocessors:
          state = preproc.process_state_for_network(state)
        return state

    def process_state_for_memory(self, state):
        """
        simply runs all the pre-processors
        """
        for preproc in self.preprocessors:
          state = preproc.process_state_for_memory(state)
        return state

    def reset(self):
      for preproc in self.preprocessors:
        preproc.reset()

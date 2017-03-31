import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add

def get_model(model_name, *args):
  if model_name == 'linear':
    return linear(*args)
  elif model_name == 'convnet':
    return convnet(*args)
  elif model_name == 'simpler_convnet':
    return simpler_convnet(*args)
  elif model_name == 'nature_convnet':
    return nature_convnet(*args)
  elif model_name == 'convnet_bn':
    return convnet_bn(*args)
  elif model_name == 'dueling_convnet':
    return dueling_convnet(*args)
  else:
    raise ValueError()

def linear(input_shape, num_actions):
  model = Sequential()
  model.add(Flatten(
    input_shape=input_shape))
  model.add(Dense(
    num_actions,
    activation=None))
  return model

def convnet(input_shape, num_actions):
  model = Sequential()
  model.add(Conv2D(16, 8, strides=(4, 4),
            activation='relu',
            input_shape=input_shape))
  model.add(Conv2D(32, 4, strides=(2, 2),
            activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(num_actions, activation=None))
  return model

def convnet_bn(input_shape, num_actions):
  model = Sequential()
  model.add(Conv2D(16, 8, strides=(4, 4),
            activation='relu',
            input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Conv2D(32, 4, strides=(2, 2),
            activation='relu'))
  model.add(BatchNormalization())
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(num_actions, activation=None))
  return model

def simpler_convnet(input_shape, num_actions):
  model = Sequential()
  model.add(Conv2D(16, 8, strides=(4, 4),
            activation='relu',
            input_shape=input_shape))
  model.add(Conv2D(32, 4, strides=(2, 2),
            activation='relu'))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(num_actions, activation=None))
  return model

def nature_convnet(input_shape, num_actions):
  model = Sequential()
  model.add(Conv2D(32, 8, strides=(4, 4),
            activation='relu',
            input_shape=input_shape))
  model.add(Conv2D(64, 4, strides=(2, 2),
            activation='relu'))
  model.add(Conv2D(64, 3, strides=(1, 1),
            activation='relu'))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(num_actions, activation=None))
  return model

def dueling_convnet(input_shape, num_actions):
  inputs = Input(shape=input_shape)
  net = Conv2D(16, 8, strides=(4, 4),
               activation='relu')(inputs)
  net = Conv2D(32, 4, strides=(2, 2),
               activation='relu')(net)
  net = Flatten()(net)
  advt = Dense(256, activation='relu')(net)
  advt = Dense(num_actions)(advt)
  value = Dense(256, activation='relu')(net)
  value = Dense(1)(value)
  # now to combine the two streams
  advt = Lambda(lambda advt: advt - tf.reduce_mean(
    advt, axis=-1, keep_dims=True))(advt)
  value = Lambda(lambda value: tf.tile(value, [1, num_actions]))(value)
  final = Add()([value, advt])
  model = Model(
    inputs=inputs,
    outputs=final)
  return model

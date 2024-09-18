# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import json
import math

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Add, Activation

from layer_normalization import LayerNormalization
from activations import *
from losses import huber_loss


input_dim = 198
action_space = 6

# Creating the network
x = Input(shape=(input_dim,))
h = Dense(units=256, activation='swish')(x)
h = Dense(units=128, activation='swish')(h)
v = Dense(units=1, activation="gelu", name='Value')(h)
a = Dense(units=action_space, activation='gelu')(h)
a = LayerNormalization(center=False, name='Advantage')(a)
y = Add(name='Q_values')([v,a])

model_name = "D3RQN_primary"
model = Model(inputs=x, outputs=y, name=model_name)

model.load_weights("./TrainedModels/DQN_ep=10000.h5")
model.summary()


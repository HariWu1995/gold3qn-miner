# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from random import random, randrange
import json
import math
import traceback

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Add, Activation
from keras.optimizers import Nadam, Adam, SGD, RMSprop

# from layer_normalization import LayerNormalization
from activations import *
from losses import huber_loss


# Deep Q Network off-policy
class Agent: 
   
    def __init__(self,
                 input_dim, # The number of inputs for the DQN network
                 action_space, # The number of actions for the DQN network
                 gamma=0.89, # The discount factor
                 epsilon=0.99, # The exploration rate
                 epsilon_min=0.01, 
                 epsilon_max=0.99,
                 epsilon_decay=0.997, # The decay speed in each episode
                 learning_rate=0.011, 
                 tau=0.89, # The likelihood for updating the DQN target network from the DQN network
                 primary_model=None, # pretrained DQN model
                 target_model=None, # pretrained DQN target model 
                 sess=None):

        self.input_dim = input_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.logger_dir = "./logs/"
            
        # Creating networks
        if not primary_model:
            # Creating the DQN primary model
            self.primary_model = self.create_model("D3QN_primary")
        else:
            self.primary_model = primary_model
        # optim = Adam(lr=self.learning_rate)
        optim = RMSprop(lr=self.learning_rate)
        # optim = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        self.primary_model.compile(optimizer=optim, loss=huber_loss)

        if not target_model:
            # Creating the DQN target model
            self.target_model = self.create_model("D3QN_target") 
        else:
            self.target_model = target_model

        # Tensorflow GPU optimization
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.logger = tf.summary.FileWriter(self.logger_dir, self.sess.graph)
        K.set_session(sess)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def create_model(self, model_name="D3QN"):
        # Creating the network
        x = Input(shape=(self.input_dim,))
        h = Dense(units=256, activation='swish')(x)
        h = Dense(units=128, activation='swish')(h)
        v = Dense(units=1, activation="gelu", name='Value')(h)
        a = Dense(units=self.action_space, activation='softmax')(h)
        y = Add(name='Q_values')([v,a])

        model = Model(inputs=x, outputs=y, name=model_name)
        return model

    def adjust_lr(self, new_lr):
        K.set_value(self.primary_model.optimizer.learning_rate, new_lr)
    
    def act(self, state):
        # Get the index of the maximum Q values      
        if random() < self.epsilon:
            action_explore = randrange(self.action_space)
            return action_explore
        state = np.expand_dims(state, axis=0)
        action_exploit = np.argmax(self.primary_model.predict(state), axis=0)[0]
        return action_exploit
    
    def replay(self, samples, batch_size):

        state = np.vstack(samples[:,0])
        action = samples[:,1]
        reward = samples[:,2]
        future_bool = samples[:,3]
        new_state = np.vstack(samples[:,4])

        Q1 = self.target_model.predict(state)
        Q2 = self.target_model.predict(new_state)
        Q2 = np.max(Q2, axis=-1)
        targets = Q1
        for i in range(batch_size):
            targets[i, action[i]] = reward[i] + self.gamma*Q2[i]*future_bool[i]

        inputs = state.reshape([batch_size, self.input_dim])
        targets = targets.reshape([batch_size, self.action_space])
        
        # Training
        loss = self.primary_model.train_on_batch(inputs, targets)
        return loss

    def update_logger(self, episode, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            summary = tf.Summary()
            summary.value.add(tag=key, simple_value=value)
            self.logger.add_summary(summary, global_step=episode)
            
    def update_target(self): 
        weights = self.primary_model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(0, len(target_weights)):
            target_weights[i] = weights[i]*self.tau + target_weights[i]*(1-self.tau)
        self.target_model.set_weights(target_weights) 
    
    def update_epsilon(self, step, total_steps):
        self.epsilon = (1-step/total_steps) ** 2
        if self.epsilon > self.epsilon_max:
            self.epsilon = self.epsilon_max
        elif self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
    
    def save_model(self, path, model_name):
        # serialize model to JSON
        model_config = self.primary_model.to_json()
        with open(path+model_name+".json", "w") as json_file:
            json_file.write(model_config)

        # serialize weights to HDF5
        self.primary_model.save_weights(path+model_name+".h5")
        # print("Saved model to disk")
 


# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import json
import math
import traceback

import numpy as np
from random import random as randprob, randrange

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Flatten, Concatenate, Add, Activation
from keras.optimizers import Nadam, Adam, SGD, RMSprop

# from layer_normalization import LayerNormalization
from activations import *
from losses import huber_loss


# Directed-Future Prediction Q-Network (off-policy)
class Agent: 
   
    def __init__(self,
                 map_size, 
                 max_players,
                 num_measures, # The number of measures to track Game-Info
                 num_goals, # The number of goals to prioritize Game
                 num_actions, # The number of actions for the DFP network
                 future_contribution=0.89, # The discount factor of future Q-value contribution
                 epsilon_strategy="adaptive", # Which strategy to adjust exploration rate
                 epsilon_init=0.89,
                 epsilon_min=0.11, 
                 epsilon_max=0.79,
                 epsilon_losing_offset=0.1, # The increasing offset when agent loses
                 epsilon_winning_coeff=0.67, # The decreasing slope when agent wins
                 learning_rate_init=0.13, 
                 learning_rate_min=1e-29, 
                 learning_rate_max=0.59, 
                 target_likelihood=0.83, # The likelihood for updating the DFP target network from the DFP network
                 primary_model=None, # pretrained DFP model
                 target_model=None, # pretrained DFP target model 
                 sess=None):

        # Model hyper-params
        self.map_size = map_size
        self.state_size = (max_players+4,) + self.map_size # 4 terrains
        self.measure_size = (num_measures,)
        self.goal_size = (num_goals,)
        self.num_actions = num_actions

        # Training hyper-params
        self.gamma = future_contribution
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_strategy = epsilon_strategy
        self.epsilon_winning_coeff = epsilon_winning_coeff
        self.epsilon_losing_offset = epsilon_losing_offset
        self.learning_rate = learning_rate_init
        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.tau = target_likelihood
        self.logger_dir = "./logs/"
            
        # Creating networks
        if not primary_model:
            # Creating the DFP primary model
            self.primary_model = self.create_model("DFP_primary")
        else:
            self.primary_model = primary_model
        # optim = Adam(lr=self.learning_rate)
        optim = RMSprop(lr=self.learning_rate)
        # optim = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        self.primary_model.compile(optimizer=optim, loss=huber_loss)

        if not target_model:
            # Creating the DFP target model
            self.target_model = self.create_model("DFP_target") 
        else:
            self.target_model = target_model

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not sess:
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess
        self.logger = tf.summary.FileWriter(self.logger_dir, self.sess.graph)
        K.set_session(sess)
        self.sess.run(tf.global_variables_initializer())

    def create_model(self, model_name="DFP"):
        # Creating the network
        space_input = Input(shape=self.state_size)
        space_feature = Flatten()(space_input)
        space_feature = Dense(units=256, activation='swish')(space_feature)
        space_feature = Dense(units=128, activation='swish')(space_feature)

        measure_input = Input(shape=self.measure_size)
        if len(self.measure_size) > 1:
            measure_feature = Flatten()(measure_input)
        else:
            measure_feature = measure_input
        measure_feature = Dense(units=32, activation='swish')(measure_feature)
        measure_feature = Dense(units=64, activation='swish')(measure_feature)

        goal_input = Input(shape=self.goal_size)
        goal_feature = Dense(units=16, activation='swish')(goal_input)
        goal_feature = Dense(units=64, activation='swish')(goal_feature)

        features = Concatenate()([space_feature, measure_feature, goal_feature])

        expectation = Dense(units=1, activation="gelu", name='Expectation')(features)
        advantage = Dense(units=self.num_actions, activation='softmax', name='Advantage')(features)
        q_value = Add(name='Q_values')([expectation, advantage])

        model = Model(inputs=[space_input, measure_input, goal_input], 
                      outputs=q_value, name=model_name)
        return model

    def adjust_lr(self, new_lr):
        new_lr = min(self.learning_rate_max, new_lr)
        new_lr = max(self.learning_rate_min, new_lr)
        self.learning_rate = new_lr
        K.set_value(self.primary_model.optimizer.learning_rate, new_lr)
    
    def act(self, state):
        agent_heatmap = state[0][4]
        heatmap_peak = np.where(agent_heatmap==np.amax(agent_heatmap))
        agent_position = (heatmap_peak[0][0], heatmap_peak[1][0])

        actions = list(range(6))
        action_probs = [1.0/self.num_actions] * self.num_actions
        action_probs = np.array(action_probs)
        if agent_position[1] == 0:
            # reduce probability of action GO_LEFT
            action_probs[0] /= self.num_actions
        elif agent_position[1] == self.map_size[1]-1:
            # reduce probability of action GO_RIGHT
            action_probs[1] /= self.num_actions
        if agent_position[0] == 0:
            # reduce probability of action GO_UP
            action_probs[2] /= self.num_actions
        elif agent_position[0] == self.map_size[0]-1:
            # reduce probability of action GO_DOWN
            action_probs[3] /= self.num_actions
        total_probs = np.sum(action_probs)
        action_probs += (1-total_probs) / self.num_actions 

        # Randomize action
        if randprob() < self.epsilon:
            action_explore = np.random.choice(actions, p=action_probs)
            return action_explore

        # Get the index of the maximum Q values      
        state_batch = [np.expand_dims(s_i, axis=0) for s_i in state]
        action_batch = self.primary_model.predict(state_batch)[0]
        action_batch += action_probs
        action_exploit = np.argmax(action_batch)
        return action_exploit
    
    def replay(self, samples, batch_size):

        state = [np.stack(samples[:,idx]) for idx in [0,1,2]]
        action = samples[:,3]
        reward = samples[:,4]
        future_bool = samples[:,5]
        next_state = [np.stack(samples[:,idx]) for idx in [6,7,8]]

        Q1 = self.target_model.predict(state)
        Q2 = self.target_model.predict(next_state)
        Q2 = np.max(Q2, axis=-1)
        targets = Q1
        for i in range(batch_size):
            targets[i, action[i]] = reward[i] + self.gamma*Q2[i]*future_bool[i]
        
        # Training
        loss = self.primary_model.train_on_batch(state, targets)
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
    
    def update_epsilon(self, **kwargs):
        if self.epsilon_strategy == "e-greedy":
            self.epsilon_greedy(kwargs['step'], kwargs['total_steps'])
        elif self.epsilon_strategy == "adaptive":
            self.epsilon_adaptive(kwargs['rank'])
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.epsilon = min(self.epsilon, self.epsilon_max)

    def epsilon_adaptive(self, rank):
        if rank < 0.07:
            self.epsilon *= self.epsilon_winning_coeff
        else:
            self.epsilon += self.epsilon_losing_offset * rank

    def epsilon_greedy(self, step, total_steps):
        if step < total_steps*0.25:
            self.epsilon = self.epsilon_max
        elif total_steps*0.25 <= step < total_steps*0.5:
            a = 2 * (self.epsilon_min-self.epsilon_max) / total_steps
            b = (3*self.epsilon_max-self.epsilon_min) / 2
            self.epsilon = a*step + b
        elif total_steps*0.5 <= step < total_steps*0.75:
            a = (6*self.epsilon_min-2*self.epsilon_max) / total_steps
            b = (3*self.epsilon_max-7*self.epsilon_min) / 2
            self.epsilon = a*step + b
        else:
            self.epsilon = self.epsilon_min
    
    def save_model(self, path, model_name):
        # serialize model to JSON
        model_config = self.primary_model.to_json()
        with open(path+model_name+".json", "w") as json_file:
            json_file.write(model_config)

        # serialize weights to HDF5
        self.primary_model.save_weights(path+model_name+".h5")
        # print("Saved model to disk")
 


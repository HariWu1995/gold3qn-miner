"""
Experience Replay
------------------------------------------------
This class allows user to store experiences 
and then, sample randomly to train the network.
"""
from glob import glob
from collections import deque

import os
import pickle
import random
import numpy as np


class ExpBuffer:
    
    def __init__(self, capacity=1024, storage='./ExpWealth/'):
    
        self.capacity = capacity
        self.ExpShape = 9
        self.storage = storage
        # self.ExpWealth = deque(maxlen=self.capacity)

        pos_wealths = glob(self.storage+'pos_wealth_*.pkl')
        if len(pos_wealths) > 0:
            self.PosWealth = self.restore(max(pos_wealths, key=os.path.getmtime))
        else:
            print("Creating ExpWealth for positive samples")
            self.PosWealth = deque(maxlen=self.capacity)

        neg_wealths = glob(self.storage+'neg_wealth_*.pkl')
        if len(neg_wealths) > 0:
            self.NegWealth = self.restore(max(neg_wealths, key=os.path.getmtime))
        else:
            print("Creating ExpWealth for negative samples")
            self.NegWealth = deque(maxlen=self.capacity)

    def restore(self, pickle_file):
        print(f"Loading {pickle_file}")
        with open(pickle_file, 'rb') as storage:
            wealth = pickle.load(storage)
        return wealth

    def store(self, suffix=0):
        pos_file = self.storage + f'pos_wealth_{suffix}.pkl'
        with open(pos_file, 'wb') as storage:
            pickle.dump(self.PosWealth, storage)

        neg_file = self.storage + f'neg_wealth_{suffix}.pkl'
        with open(neg_file, 'wb') as storage:
            pickle.dump(self.NegWealth, storage)

    def __len__(self):
        # return len(self.ExpWealth)
        return min(len(self.PosWealth), len(self.NegWealth)) * 2

    def push(self, exp_piece: tuple or list):
        """
        Experience: 
            <current_state, current_measures, current_goals, 
             action, reward, game_status, 
             next_state, next_measures, next_goals>
        """
        # self.ExpWealth.append(np.asarray(exp_piece))
        if exp_piece[4] >= 0:
            self.PosWealth += [np.asarray(exp_piece)]
        elif exp_piece[4] < 0:
            self.NegWealth += [np.asarray(exp_piece)]
            
    def sample(self, batch_size=16):
        # batch_samples = random.sample(self.ExpWealth, batch_size)
        # batch_samples = np.array(batch_samples)
        # return batch_samples.reshape([batch_size, self.ExpShape])
        positive_samples = random.sample(self.PosWealth, batch_size//2)
        positive_samples = np.array(positive_samples)
        negative_samples = random.sample(self.NegWealth, batch_size-batch_size//2)
        negative_samples = np.array(negative_samples)
        batch_samples = np.vstack([positive_samples, negative_samples])
        return batch_samples #.reshape([batch_size, self.ExpShape])
                    

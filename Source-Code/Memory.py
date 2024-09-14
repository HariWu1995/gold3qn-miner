import random
import numpy as np
from collections import deque


class Memory:
    
    def __init__(self, capacity=1024, state_size=198):
    
        self.capacity = capacity
        self.state_size = state_size # for checking shape
        self.ExpShape = 5
        self.ExpWealth = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.ExpWealth)

    def push(self, exp_piece: list):
        """
        Experience: 
            <current_state, action, reward, game_status, next_state>
        """
        assert len(exp_piece[0])==self.state_size, "Size of state is not matched"
        assert len(exp_piece[-1])==self.state_size, "Size of state is not matched"
        self.ExpWealth.append(np.asarray(exp_piece))
            
    def sample(self, batch_size=1):
        batch_samples = random.sample(self.ExpWealth, batch_size)
        batch_samples = np.array(batch_samples)
        return batch_samples.reshape([batch_size, self.ExpShape])
                    

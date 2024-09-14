import sys

import random
import numpy as np

from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State


TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:

    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score # Storing the last score for designing the reward function

    def start(self): 
        # connect to server
        self.socket.connect()

    def end(self): 
        # disconnect server
        self.socket.close()

    def send_map_info(self, request):
        # tell server which map to run
        self.socket.send(request)

    def reset(self): 
        # start new game
        try:
            message = self.socket.receive() # receive game info from server
            self.state.init_state(message) # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): 
        # step process
        self.socket.send(action) # send action to server
        try:
            message = self.socket.receive() # receive new state from server
            self.state.update_state(message) # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x+1, 
                         self.state.mapInfo.max_y+1], dtype=int)
        for i in range(self.state.mapInfo.max_x+1):
            for j in range(self.state.mapInfo.max_y+1):
                if self.state.mapInfo.get_obstacle(i,j) == TreeID:  # Tree
                    view[i,j] = -TreeID
                if self.state.mapInfo.get_obstacle(i,j) == TrapID:  # Trap
                    view[i,j] = -TrapID
                if self.state.mapInfo.get_obstacle(i,j) == SwampID: # Swamp
                    view[i,j] = -SwampID
                if self.state.mapInfo.gold_amount(i,j) > 0:
                    view[i,j] = self.state.mapInfo.gold_amount(i,j)

        # Flattening the map matrix to a vector
        DQNState = view.flatten().tolist() 

        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)

        # Add position of other player(s)
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
                
        # Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            # If the DQN agent crafts golds, 
            # then it should obtain a positive reward (=score_action)
            reward += score_action
            
        # If the DQN agent crashs into obstacles (Tree, Trap, Swamp), 
        # then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  
            reward -= random.randrange(5,20)
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID: 
            reward -= 10
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID: 
            reward -= random.choice([5,20,40,100])
            
        # If out of energy, 
        # the agent should be punished by a larger negative reward.
        if self.state.status == State.STATUS_ELIMINATED_INVALID_ACTION:
            reward -= 100

        # If practise an invalid action, 
        # the agent should be punished by a large negative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward -= 150

        # If out of the map, 
        # the agent should be punished by the largest negative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 200

        # print ("reward:", reward)
        return reward

    def check_terminate(self): 
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING


import sys

import random
import numpy as np

from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State


MineID = 0
TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:

    def __init__(self, host, port):
        self.state = State()
        self.socket = GameSocket(host, port)
        self.last_map = None
        self.last_score = self.state.score # Storing the last score for designing the reward function
        self.init_golds = None

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
        message = self.socket.receive() # receive game info from server
        self.state.init_state(message) # init state
        self.init_golds = None
        self.historic_maps = []

    def step(self, action): 
        # step process
        self.socket.send(action) # send action to server
        message = self.socket.receive() # receive new state from server
        self.state.update_state(message) # update to local state

    # Functions are customized by client
    def get_state(self):
        """
        Split each map into 8 layers:
            4 layers for terrains: Tree, Trap, Swamp & Gold-Mine
            4 layers for players' positions

        3+1 measures: 
            score/golds-mined, golds-left, energy, <current_step>

        4+1 goals: (corresponding surjectively to 3+1 measures above)
            score-rank, golds-ratio, energy-rank, energy-scale, <steps-ratio>
        """
        map_size = (self.state.mapInfo.max_x+1, self.state.mapInfo.max_y+1)
        map_state = np.zeros(shape=(8,)+map_size)
        self.historic_maps += [np.zeros(map_size)]

        # Building the map
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                if self.state.mapInfo.get_obstacle(i,j) == TreeID:
                    map_state[TreeID,i,j] = 20 #random.randrange(5,20)
                elif self.state.mapInfo.get_obstacle(i,j) == TrapID:
                    map_state[TrapID,i,j] = 10
                elif self.state.mapInfo.get_obstacle(i,j) == SwampID:
                    map_state[SwampID,i,j] = 70 #random.choice([5,20,40,100])
                elif self.state.mapInfo.gold_amount(i,j) > 0:
                    map_state[MineID,i,j] = self.state.mapInfo.gold_amount(i,j)

        for obstacleID in [TreeID, TrapID, SwampID]:
            self.historic_maps[-1] += map_state[obstacleID,:,:]

        if not self.init_golds:
            self.init_golds = np.sum(map_state[MineID,:,:])
        golds_left = np.sum(map_state[MineID,:,:])

        # Create heatmap for self-player
        map_state[4,:,:] = get_heatmap(map_size, [self.state.x, self.state.y])
        players_score = [self.state.score]
        players_energy = [self.state.energy]
        player_measures = np.array(
            [self.state.score, golds_left, self.state.energy, 1e-7], dtype=float)

        # Create heatmap(s) for other player(s)
        player_id = 5
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                map_state[player_id,:,:] = get_heatmap(map_size, [player["posx"], player["posy"]])
                players_score += [player["score"] if "score" in player.keys() else self.state.score]
                players_energy += [player["energy"] if "energy" in player.keys() else self.state.energy]
                player_id += 1
                
        # Rank player's score / energy vs. other players
        player_score_rank = np.where(np.argsort(players_score)==0)[0][0] + 1
        player_energy_rank = np.where(np.argsort(players_energy)==0)[0][0] + 1
        player_goals = np.array([
            1-player_score_rank/len(players_score), # the lowers player's score-rank is, the larger the force is to push player to mine more golds
            golds_left/self.init_golds, # the lower the amount of golds is, the faster player must find gold-mine and mine
            1-player_energy_rank/len(players_energy), # the lower player's energy-rank is, the larger the force is to push player to rest
            1-self.state.energy/50, # 50 is the init energy (if prioritize safety, choose 101 because -100 is the largest penalty)
            1e-7
        ])

        return (map_state, player_measures, player_goals)

    def get_reward(self, state, next_state):

        heatmap, next_heatmap = state[0], next_state[0]
        ranks, next_ranks = state[2], next_state[2]

        # Calculate reward
        reward = 0
        golds_mined = self.state.score - self.last_score
        if golds_mined > 0:
            reward += golds_mined
        elif ranks[1] > next_ranks[1]:
            golds_mined_by_others = (ranks[1]-next_ranks[1]) * self.init_golds
            reward -= golds_mined_by_others // 10
        self.last_score = self.state.score
            
        # If the agent crashs into an obstacle (Tree/Trap/Swamp), 
        # it should be punished by a penalty
        for obstacleID in [TreeID, TrapID, SwampID]:
            if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == obstacleID:
                # use the 2nd last map because the last map is of next state
                reward -= self.historic_maps[-2][self.state.x, self.state.y] 

        # If go closer to gold-mine, agent should get a reward.
        # Otherwise, be punished.
        position_similarity = np.sum(heatmap[0]*heatmap[4])
        next_position_similarity = np.sum(heatmap[0]*next_heatmap[4])
        if golds_mined <= 0:
            reward += (next_position_similarity-position_similarity) * 10

        # If out of energy, 
        # the agent should be punished by a large negative reward.
        if self.state.status == State.STATUS_ELIMINATED_INVALID_ACTION:
            reward -= 100

        # If practise an invalid action, 
        # the agent should be punished by a larger negative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward -= 150

        # If out of the map, 
        # the agent should be punished by the largest negative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 200

        # If agent ranks up in score / energy, he should be rewarded
        # Otherwise ranking down, penalty should be applied
        reward += (ranks[0]-next_ranks[0]) * 200 # score
        reward += (ranks[2]-next_ranks[2]) * 100 # energy

        # In case nearly out of golds or time,
        # reward / penalty should be double
        if ranks[4] > 0.89 or next_ranks[1] < 0.11:
            reward *= 2

        # print ("reward:", reward)
        return int(reward)

    def check_terminate(self): 
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING


def get_heatmap(shape_2d, pos, scale=1.0, epsilon: float=1e-7) -> np.array:
    heatmap = np.ones(shape_2d)

    for i in range(heatmap.shape[0]):
        heatmap[i,:] -= abs(i-pos[0])

    for i in range(heatmap.shape[1]):
        heatmap[:,i] -= abs(i-pos[1])

    heatmap /= np.prod(heatmap.shape) + epsilon
    return heatmap*scale

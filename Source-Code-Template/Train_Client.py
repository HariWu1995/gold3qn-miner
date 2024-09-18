import sys
import datetime 
import traceback

from tqdm import tqdm
from statistics import mean, median

from Agent import Agent # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process

import pandas as pd
import numpy as np


def moving_average(x, N=3):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:]-cumsum[:-N]) / float(N)


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() # Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", 
          "Action", "Epsilon", "Done", "Termination_Code"] # Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
INIT_EPISODE = 1060000
END_EPISODE = 1200000 # The number of episodes for training
N_EPISODES = END_EPISODE - INIT_EPISODE
MAX_STEP = 1000 # The number of steps for each episode
BATCH_SIZE = 5678 # The number of experiences for each replay 
MEMORY_SIZE = 1024098 # The size of the batch for storing experiences
SAVE_NETWORK = N_EPISODES//10 # After this #episodes, save model for testing later. 
UPDATE_TARGET = 256
MIN_REPLAY_SIZE = BATCH_SIZE * 4 # The number of experiences are stored in the memory batch before starting replaying
ACTION_SIZE = 6 # The number of actions output from the DQN model
MAP_MAX_X = 21 # Width of the Map
MAP_MAX_Y = 9 # Height of the Map
MAP_AREA = MAP_MAX_X * MAP_MAX_Y
INPUT_SIZE =  MAP_AREA + 9 # The number of input values for the DQN model
                           # 189: area of map + (posX, posY, E) + 3x(posX, posY)
LOSS_SATURATION = 128
LOSS_ELASTICITY = 7
LOSS_CHAIN_LEN = 16
loss_recorder = []

# Initialize a DQN model and a memory batch for storing experiences
Agent = Agent(INPUT_SIZE, ACTION_SIZE)
Agent.primary_model.load_weights("TrainedModels/DQN_ep=1040000.h5")
Agent.target_model.load_weights("TrainedModels/DQN_ep=1060000.h5")

memory = Memory(MEMORY_SIZE, state_size=INPUT_SIZE)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) # Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game


# Training Process: the main part of the deep-Q learning agorithm 
for episode_i in tqdm(range(INIT_EPISODE, END_EPISODE)):
    
    # Choosing a map in the list
    mapID = np.random.randint(1, 6) # Choosing a map ID from 5 maps in Maps folder randomly
    posID_x = np.random.randint(MAP_MAX_X) # Choosing a initial position of the DQN agent on X-axes randomly
    posID_y = np.random.randint(MAP_MAX_Y) # Choosing a initial position of the DQN agent on Y-axes randomly
    
    # Creating a request for initializing a map, 
    # initial position, the initial energy & the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    
    # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)

    # Getting the initial state
    minerEnv.reset() # Initialize the game environment
    s = minerEnv.get_state() # Get the state after reseting. 
    total_reward = 0 # The amount of rewards for the entire episode
    terminate = False # The variable indicates that the episode ends
    maxStep = minerEnv.state.mapInfo.maxStep # Get the maximum number of steps for each episode in training
    
    # Start an episde for training
    for step in range(0, maxStep):
        action = Agent.act(s)  # Getting an action from the DQN model from the state (s)
        minerEnv.step(str(action))  # Performing the action in order to obtain the new state
        s_next = minerEnv.get_state()  # Getting a new state
        reward = minerEnv.get_reward()  # Getting a reward
        terminate = minerEnv.check_terminate()  # Checking the end status of the episode

        # Add this transition to the memory batch
        exp_piece = [s, action, reward, terminate, s_next]
        memory.push(exp_piece)

        total_reward += reward # Add current reward to the total reward of the episode
        s = s_next # Assign the next state for the next step.

        # Saving data to file
        # save_data = np.hstack(
        #     [episode_i+1, step+1, reward, total_reward, action, Agent.epsilon, terminate])
        # save_data = save_data.reshape(1,7)
        # with open(filename, 'a') as f:
        #     pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
        
        if terminate == True:
            # If the episode ends, then go to the next episode
            break

    # Only start replaying when memory has enough experience
    if len(memory) < MIN_REPLAY_SIZE:
        continue

    # Sample batch memory to train network
    batch = memory.sample(BATCH_SIZE)
    loss = Agent.replay(batch, BATCH_SIZE)

    # Adjust learning rate
    loss_recorder += loss
    if len(loss_recorder) > LOSS_CHAIN_LEN:
        loss_recorder = loss_recorder[1:]
        # loss_mean = mean(loss_recorder)
        loss_avgs = moving_average(loss_recorder)
        loss_trend = loss_avgs >= 0
        loss_max = max(loss_recorder)
        loss_min = min(loss_recorder)
        old_lr = Agent.primary_model.optimizer.learning_rate
        if 0.41 < np.sum(loss_trend)/len(loss_recorder) < 0.59:
            new_lr = old_lr * 2
        elif loss_max / (loss_min+1e-11) > LOSS_ELASTICITY:
            new_lr = old_lr ** 2
        else:
            new_lr = old_lr
        Agent.adjust_lr(min(0.97, new_lr))

    # Replace the weights of target model with soft replacement
    if np.mod(episode_i+1, UPDATE_TARGET) == 0:
        Agent.update_target()

    # Iteration to save the network architecture and weights
    if np.mod(episode_i+1, SAVE_NETWORK) == 0:
        # now = datetime.datetime.now() # Get the latest datetime
        Agent.save_model("TrainedModels/", f"DQN_ep={episode_i+1}")
    
    # Record the training information after the episode
    train_logs = {
        "loss": loss,
        "n_golds": minerEnv.state.score,
        "n_steps": step+1,
        "reward_avg": total_reward//(step+1), 
        "terminate_code": minerEnv.state.status,
        "exploration_rate": Agent.epsilon,
    }
    Agent.update_logger(episode_i+1, train_logs)
    # print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %d. Epsilon = %.2f .Termination code: %d' % (
    #     episode_i+1, step+1, total_reward, Agent.epsilon, terminate))
    
    # Decreasing the epsilon if the replay starts
    Agent.update_epsilon(episode_i-INIT_EPISODE, END_EPISODE-INIT_EPISODE)


Agent.save_model("TrainedModels/", "DQN_"+now.strftime("%Y%m%d-%H%M"))


import sys, os, gc
import datetime 
import traceback

from tqdm import tqdm
from statistics import mean, median

from Agent import Agent # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DFP model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from ExpBuffer import ExpBuffer # A class of creating a batch in order to store experiences for the training process

import pandas as pd
import numpy as np
import random


def moving_average(x, N=3):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:]-cumsum[:-N]) / float(N)


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])


# Parameters for training a DFP model
INIT_EPISODE = 340000
END_EPISODE = 440000 # The number of episodes for training
N_EPISODES = END_EPISODE - INIT_EPISODE
MAX_STEP = 1000 # The number of steps for each episode
BATCH_SIZE = 2048 # The number of experiences for each replay 
MEMORY_SIZE = 8192 # The size of the batch for storing experiences
SAVE_NETWORK = N_EPISODES//10 # After this #episodes, save model for testing later. 
UPDATE_TARGET = 128
MIN_REPLAY_SIZE = BATCH_SIZE*2 # The number of experiences are stored in the memory batch before starting replaying

MAP_MAX_X = 21 # Width of the Map
MAP_MAX_Y = 9 # Height of the Map
ACTION_SIZE = 6 # The number of actions output from the DFP model

# LOSS_SATURATION = 128
LOSS_ELASTICITY = 0.97
LOSS_STABILITY = 0.59
LOSS_EXTREMES = 0.79
LOSS_CHAIN_LEN = 1024
loss_recorder = []
UPDATE_LR = 789

# Initialize a DFP model and a memory batch for storing experiences
Agent = Agent(map_size=(MAP_MAX_X, MAP_MAX_Y), 
              max_players=4,
              num_measures=4,
              num_goals=5,
              num_actions=ACTION_SIZE)
# Agent.primary_model.load_weights("TrainedModels/DFP_ep=320000.h5")
# Agent.target_model.load_weights("TrainedModels/DFP_ep=340000.h5")

expBuffer = ExpBuffer(MEMORY_SIZE)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) # Creating a communication environment between the DFP model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game


# Training Process: the main part of the deep-Q learning agorithm 
for episode_i in tqdm(range(INIT_EPISODE, END_EPISODE)):
    
    # Choosing a map in the list
    mapID = np.random.randint(1, 6) # Choosing a map ID from 5 maps in Maps folder randomly
    posID_x = np.random.randint(MAP_MAX_X) # Choosing a initial position of the DFP agent on X-axes randomly
    posID_y = np.random.randint(MAP_MAX_Y) # Choosing a initial position of the DFP agent on Y-axes randomly
    
    # Creating a request for initializing a map, 
    # initial position, the initial energy & the maximum number of steps of the DFP agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    
    # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)

    # Getting the initial state
    minerEnv.reset() # Initialize the game environment
    maxStep = minerEnv.state.mapInfo.maxStep # Overwrite the maximum number of steps for each episode in training
    s = minerEnv.get_state() # Get the state after reseting. 
    s[1][-1] = 1 # Add step index to measures
    s[2][-1] = 1/maxStep
    total_reward = 0 # The amount of rewards for the entire episode
    terminate = False # The variable indicates that the episode ends
    ranking = -1
    
    # Start an episde for training
    for step in range(0, maxStep):
        action = Agent.act(s)  # Get an action from the DFP model from the state
        minerEnv.step(str(action))  # Practice the action in order to obtain the new state
        s_next = minerEnv.get_state()  # Gett a new state
        s_next[1][-1] = step+1
        s_next[2][-1] = (step+1)/maxStep
        reward = minerEnv.get_reward(s, s_next)  # Get a reward
        terminate = minerEnv.check_terminate()  # Check the status of the episode

        # Add this transition to the memory batch
        exp_piece = s + (action, reward+step, terminate) + s_next
        expBuffer.push(exp_piece)

        total_reward += reward # Add current reward to the total reward of the episode
        s = s_next # Assign the next state for the next step.

        # Saving data to file
        # save_data = np.hstack(
        #     [episode_i+1, step+1, reward, total_reward, action, DFPAgent.epsilon, terminate])
        # save_data = save_data.reshape(1,7)
        # with open(filename, 'a') as f:
        #     pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
        
        if terminate == True:
            # If the episode ends, then go to the next episode
            break

    # Only start replaying when memory has enough experience
    if len(expBuffer) < MIN_REPLAY_SIZE:
        continue

    # Sample batch memory to train network
    batch = expBuffer.sample(BATCH_SIZE)
    loss = Agent.replay(batch, BATCH_SIZE) 

    # Adjust learning rate
    loss_recorder += [loss]
    if len(loss_recorder) > LOSS_CHAIN_LEN: 
        loss_recorder = loss_recorder[-LOSS_CHAIN_LEN:]
        if np.mod(episode_i+1, UPDATE_LR) == 0:
            losses = np.array(loss_recorder)
            loss_mean = mean(loss_recorder)
            loss_max = max(loss_recorder) 
            loss_min = min(loss_recorder)
            # loss_avgs = moving_average(loss_recorder)
            upper_lim = loss_max * 0.89
            lower_lim = loss_min * 0.11
            # loss_trend = loss_avgs >= 0
            n_losses_too_high = np.sum(losses>upper_lim)
            n_losses_too_low = np.sum(losses<lower_lim)
            n_losses_extremes = (n_losses_too_high+n_losses_too_low)
            n_losses_stable = np.sum(np.abs(losses-loss_mean)<loss_mean*0.11)

            old_lr = Agent.learning_rate
            if n_losses_extremes / len(loss_recorder) > LOSS_EXTREMES:
                new_lr = old_lr / 2
            elif n_losses_stable / len(loss_recorder) > LOSS_STABILITY:
                new_lr = old_lr * 5
            elif (loss_max-loss_min) > LOSS_ELASTICITY:
                new_lr = old_lr / 3
            else:
                lr_variance = Agent.learning_rate_min
                new_lr = old_lr + random.uniform(-lr_variance, lr_variance)
            Agent.adjust_lr(new_lr)

    # Replace the weights of target model with soft replacement
    if np.mod(episode_i+1, UPDATE_TARGET) == 0:
        Agent.update_target()
        UPDATE_TARGET = int(UPDATE_TARGET*1.1)

    # Iteration to save the network architecture and weights
    if np.mod(episode_i+1, SAVE_NETWORK) == 0:
        # now = datetime.datetime.now() # Get the latest datetime
        Agent.save_model("TrainedModels/", f"DFP_ep={episode_i+1}")
        expBuffer.store(episode_i+1)
    
    # Record the training information after the episode
    train_logs = {
        "loss": loss,
        "n_golds": minerEnv.state.score,
        "n_steps": step+1,
        "ranking": 1-s_next[2][0],
        "reward_avg": total_reward//(step+1), 
        "learning_rate": Agent.learning_rate,
        "terminate_code": minerEnv.state.status,
        "exploration_rate": Agent.epsilon
    }
    Agent.update_logger(episode_i+1, train_logs)
    
    # Decreasing the epsilon if the replay starts
    # Agent.update_epsilon(step=episode_i-INIT_EPISODE, total_steps=END_EPISODE-INIT_EPISODE)
    Agent.update_epsilon(rank=s_next[2][0])

    _ = gc.collect()


Agent.save_model("TrainedModels/", "DFP_"+now.strftime("%Y%m%d-%H%M"))


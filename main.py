import numpy as np
import time
import pandas as pd
import env 
from agent import Agent
from collections import deque
from utils import load_data, setup_logger , keep_awake

def main():
    #load data
    logger = setup_logger()
    logger.info(f"Loading data ...")
    data_fp = "data.csv"  # data file path
    factor_fp = "factors.csv" 
    data = load_data(data_fp, factor_fp)
    logger.info(f"Data \n{data.head()}")

    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    # estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20240101", format="%Y%m%d"
    ):
        logger.info(f"Date : {starting + pd.DateOffset(counter)}")
        cutoff = [
            starting,
            starting
            + pd.DateOffset(
                years=8 + counter
            ),  # use 8 years and expanding as the training set
            starting
            + pd.DateOffset(
                years=10 + counter
            ),  # use the next 2 years as the validation set
            starting + pd.DateOffset(years=11 + counter),
        ]  # use the next year as the out-of-sample testing set

        # cut the sample into training, validation, and testing sets
        training_set = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate_set = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test_set  = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]
        
        # create the env
        env = env.MarketEnvironment(training_set)
        agent = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(), random_seed=0)
        




# env = sca.MarketEnvironment(data) 
# cur_state = env.get_state()
# print(cur_state)

# Initialize Feed-forward DNNs for Actor and Critic models. 
# agent = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(), random_seed=0)

# Set the liquidation time
lqt = 60

# Set the number of trades
n_trades = 60

# Set trader's risk aversion
tr = 1e-6

# Set the number of episodes to run the simulation
episodes = 10000

shortfall_hist = np.array([])
shortfall_deque = deque(maxlen=100)

def train(env, agent, dataset):
    for episode in keep_awake(range(episodes)): 

        cur_state = env.reset(seed = episode, liquid_time = lqt, num_trades = n_trades, lamb = tr)

        # set the environment to make transactions
        env.start_transactions()

        for i in range(n_trades + 1):
        
            # Predict the best action for the current state. 
            action = agent.act(cur_state, add_noise = True)
            
            # Action is performed and new state, reward, info are received. 
            new_state, reward, done, info = env.step(action)
            
            # current state, action, reward, new state are stored in the experience replay
            agent.step(cur_state, action, reward, new_state, done)
            
            # roll over new state
            cur_state = new_state

            if info.done:
                shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
                shortfall_deque.append(info.implementation_shortfall)
                break
            
        if (episode + 1) % 10 == 0: # print average shortfall over last 100 episodes
            print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))        

# print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(np.mean(shortfall_hist)))

main()
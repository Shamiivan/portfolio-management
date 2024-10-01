import numpy as np
import time
import pandas as pd
import env 
from typing import List

# from agent import Agent
from collections import deque
import sys
import os
from sklearn.preprocessing import StandardScaler
from ppo import PPO
from collections import namedtuple
from env import MarketEnv


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])

TRAIN_WINDOW_SIZE = 4
VALID_WINDOW_SIZE = 2
TEST_WINDOW_SIZE = 2
PRED_WINDOW_SIZE = TRAIN_WINDOW_SIZE + VALID_WINDOW_SIZE + TEST_WINDOW_SIZE
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from custom_utils import get_logger

def main():
    #load data
    logger = get_logger("./logs/rl.log")
    logger.info(f"Loading data ...")
    
    # data_fp = "./dataset/data.csv"  
    data_fp = "./dataset/dev.csv"  
    factor_fp = "./dataset/factors.csv" 

    raw_data = pd.read_csv(data_fp, parse_dates=["date"], low_memory=False)
    stock_vars = list(pd.read_csv(factor_fp)["variable"].values)

    ret_var = "stock_exret"    
    # Filter rows where the response variable (ret_var) is not missing (NaN)
    valid_rows = raw_data[ret_var].notna()
    # Create a copy of the filtered data 
    new_set = raw_data[valid_rows].copy()  
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    #rank the data by month 
    data = process_monthly_data(monthly, stock_vars)  

    # starting : Timestamp obj
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()
    # estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20240101", format="%Y%m%d"
    ):
        cutoff = [
            starting, starting + pd.DateOffset( years=8 + counter), 
            starting + pd.DateOffset(years=10 + counter),  
            starting + pd.DateOffset(years=11 + counter),
        ]  

        # cut the sample into training, validation, and testing sets
        training_set = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate_set = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test_set  = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        train, validate, test = standardize(training_set, validate_set, test_set, stock_vars)

        # train np.ndarray: A NumPy array of shape (n_samples, n_features) representing the training data.
        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values
        train_window(train, stock_vars, ret_var)
        
        


# Function to rank transform each variable for a given group of data
def rank_transform(group: pd.DataFrame, stock_vars: list) -> pd.DataFrame:
    """
    Apply rank transformation to each variable in stock_vars for the given group.
    
    Parameters:
    group (pd.DataFrame): The group of data for a specific date.
    stock_vars (list): The list of variables to be rank-transformed.

    Returns:
    pd.DataFrame: Transformed data with variables scaled to the range [-1, 1].
    """
    for var in stock_vars:
        var_median = group[var].median(skipna=True)  # Get the median value of the variable
        group[var] = group[var].fillna(var_median)  # Fill missing values with the median

        # Rank transform the variable to [-1, 1]
        group[var] = group[var].rank(method="dense") - 1
        group_max = group[var].max()
        
        if group_max > 0:
            group[var] = (group[var] / group_max) * 2 - 1  # Rescale to [-1, 1]
        else:
            group[var] = 0  # Handle all missing values
            print(f"Warning: {group['date'].iloc[0]} {var} set to zero.")
    
    return group

#  process each monthly group the data
def process_monthly_data(monthly: pd.core.groupby.generic.DataFrameGroupBy, stock_vars: list) -> pd.DataFrame:
    """
    Process monthly data groups, apply rank transformation, and combine the results.
    
    Parameters:
    monthly (pd.core.groupby.generic.DataFrameGroupBy): Grouped data by 'date'.
    stock_vars (list): List of variables to transform.

    Returns:
    pd.DataFrame: Combined DataFrame after transformation.
    """
    transformed_data = []  # List to store transformed monthly groups

    for date, monthly_raw in monthly:
        group = monthly_raw.copy()  # Create a copy of the current monthly data group
        group = rank_transform(group, stock_vars)  # Apply the rank transformation
        transformed_data.append(group)  # Store the transformed group

    # Concatenate all transformed groups into a single DataFrame
    return pd.concat(transformed_data, ignore_index=True)

 # standardize the datasets (train, validate, test)
def standardize(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame, stock_vars: list) -> tuple:
    """
    Standardizes the train, validate, and test sets based on the training data's mean and standard deviation.
    
    Parameters:
    train (pd.DataFrame): Training dataset.
    validate (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.
    stock_vars (list): List of predictor variable names to be standardized.

    Returns:
    tuple: Standardized train, validate, and test datasets.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler().fit(train[stock_vars])  # Fit only on the training data

    # Apply the transformation to the train, validation, and test datasets
    train[stock_vars] = scaler.transform(train[stock_vars])  # Standardize training set
    # validate[stock_vars] = scaler.transform(validate[stock_vars])  # Standardize validation set
    # test[stock_vars] = scaler.transform(test[stock_vars])  # Standardize test set
    
    return train, validate, test

# Example usage:


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

def keep_awake(iterable):
    for item in iterable:
        yield item
        time.sleep(0.1) 

def train_window(train:np.array , stock_vars:list, ret_var):
    env = MarketEnv(train, stock_vars, ret_var)
    
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    print(num_state, num_action)
    exit(1)
    agent = PPO(num_state, num_action)
    
    num_episodes = 1000
    gamma = 0.9
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step([action])
            transition = Transition(state, action, reward, action_log_prob, next_state)
            if agent.store_transition(transition):
                agent.update(gamma)
            state = next_state
            episode_reward += reward
        
        print(f'Episode {episode+1}/{num_episodes}, Reward: {episode_reward}')        
# def train(env, agent, dataset):
#     for episode in keep_awake(range(episodes)): 

#         cur_state = env.reset(seed = episode, liquid_time = lqt, num_trades = n_trades, lamb = tr)

#         # set the environment to make transactions
#         env.start_transactions()

#         for i in range(n_trades + 1):
        
#             # Predict the best action for the current state. 
#             action = agent.act(cur_state, add_noise = True)
            
#             # Action is performed and new state, reward, info are received. 
#             new_state, reward, done, info = env.step(action)
            
#             # current state, action, reward, new state are stored in the experience replay
#             agent.step(cur_state, action, reward, new_state, done)
            
#             # roll over new state
#             cur_state = new_state

#             if info.done:
#                 shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
#                 shortfall_deque.append(info.implementation_shortfall)
#                 break
            
#         if (episode + 1) % 10 == 0: # print average shortfall over last 100 episodes
#             print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))        

# print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(np.mean(shortfall_hist)))

if __name__ == "__main__":
    main()
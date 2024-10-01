import gym
from gym import spaces
import numpy as np

class MarketEnv(gym.Env):

    def __init__(self, data, stock_vars, ret_var):
        super(MarketEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.stock_vars = stock_vars
        self.ret_var = ret_var
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(stock_vars),), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.data.iloc[self.current_step][self.stock_vars].values
        return obs.astype(np.float32)
    
    def step(self, action):
        # Execute one time step within the environment
        actual_return = self.data.iloc[self.current_step][self.ret_var]
        predicted_return = action[0]  
        reward = - (actual_return - predicted_return) ** 2
        
        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        info = {'actual_return': actual_return, 'predicted_return': predicted_return}
        
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        pass

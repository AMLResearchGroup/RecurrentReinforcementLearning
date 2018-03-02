import gym
from gym import spaces
from base_task import BaseTask

class limitedCartPole(BaseTask):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        
        self.mask = [0,2]
        olow  = self.env.observation_space.low[self.mask]
        ohigh = self.env.observation_space.high[self.mask]
        
        self.observation_space = spaces.Box(olow, ohigh)
        self.action_space = spaces.Discrete(2)
        
    def step(self, *args, **kwargs):
        ns, r, d, i = self.env.step(*args, **kwargs)
        ns = ns[self.mask]
        return ns, r, d, i
    
    def reset(self):
        s = self.env.reset()
        s = s[self.mask]
        return s
    
    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)
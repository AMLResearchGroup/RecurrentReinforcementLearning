
class BaseAgent(object):
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        pass
    
    def act(self, state, testing):
        raise NotImplementedError("{} must override act()".format(self.__class__.__name__))
        
    def learn(self, state, action, reward, new_state, done, testing):
        raise NotImplementedError("{} must override learn()".format(self.__class__.__name__))
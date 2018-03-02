
class BaseTask(object):
    def __init__(self):
        pass
        
    def reset():
        pass
    
    def step(self, *args, **kwargs):
        raise NotImplementedError("{} must override act()".format(self.__class__.__name__))
        
    def render(self, *args, **kwargs):
        raise NotImplementedError("{} must override learn()".format(self.__class__.__name__))
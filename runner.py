import numpy as np
import matplotlib.pyplot as plt
from tasks import BaseTask
from agents import BaseAgent

def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]
def get_subclass(cls, name):
    """Return a concrete subclass by name (classmethod)."""
    types = {t.__name__: t for t in all_subclasses(cls)}
    assert name in types, "Unknown type '{}' (must be a subclass of {})".format(name, cls.__name__)
    return types[name]

class TestRunner:
    def __init__(self, env, agent, num_episodes=1000, episode_len=None,
                 report_interval=10, test_samples=5, print_test=False,
                 runs=1):
        
        if (isinstance(env, str)):
            # Look up the env from the tasks
            env = get_subclass(BaseTask, env)()

        if (isinstance(agent, str)):
            # Look up and make the agent from agents folder
            agent = get_subclass(BaseAgent, agent)(env)
        
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.report_interval = report_interval
        self.test_samples = test_samples
        self.print_test = print_test
        self.runs = runs
        
        self.env = env
        self.agent = agent
        
        if self.report_interval is not None:
            n_reports = num_episodes // report_interval
            if (num_episodes % report_interval != 0):
                n_reports += 1
            self.tracker= np.zeros([n_reports, runs])
            self.X = np.zeros(n_reports)
    
    # This preps an "outer run" 
    def initialize_run(self):
        self.episode = -1
        self.report_index = 0
        self.testing = False
        self.cul_reward = 0
        self.run_reward = 0
        self.test_count = 0
        
        self.agent.reset()
        
    # This Function is responsible for managing the counting of episodes,
    # recording stats and determining if its a testing run
    # NOTE: THIS RUNS AT THE END of an episode. so self.testing is set for the NEXT episode
    def reset(self):
        self.cul_reward += self.run_reward
        self.run_reward = 0
        self.test_count += 1
        
        printmode = "Training"

        # If the testing counter has run out, Turn off testing mode
        if (self.test_count >= self.test_samples):
            printmode = "Testing"
            self.testing = False
            self.tracker[self.report_index, self.run] = self.cul_reward/self.test_count
            self.X[self.report_index] = self.episode + 1
            self.report_index += 1
        
        # When not in testing mode run the stats printer
        # also always reset the test count
        if (not self.testing): 
            self.episode += 1
            if (not self.print_test or printmode == "Testing"):
                print ("Episode {0:4} - {1:8} : {2:8.2f}".format(self.episode, printmode, self.cul_reward/self.test_count))
            self.cul_reward = 0
            self.test_count = 0
            
        # Turn on testing mode for the next episode if warranted
        if (self.report_interval is not None and self.episode != 0):
            next_epi = self.episode + 1
            self.testing = (next_epi % self.report_interval == 0) or (next_epi == self.num_episodes)
 
        self.state = self.env.reset()
    
    # This handles the changing of state and communication between agent and environment
    def start(self):
        for i in range(self.runs):
            print ("====================================================")
            print ("Starting Run: {}".format(i))
            # Call agent reset at start of a new round of testing
            self.initialize_run()
            self.run = i
            
            state = self.env.reset()
            
            while (self.episode < self.num_episodes):
                action = self.agent.act(state, self.testing)

                #TODO Sanitize action
                new_state, reward, done, info = self.env.step(action)

                self.run_reward += reward

                self.agent.learn(state, action, reward, new_state, done, self.testing)

                state = new_state

                if (done):                    
                    self.reset()
                
    def display(self):
        state = self.env.reset()
        done = False
        while (not done):
            self.env.render()
            action = self.agent.act(state, True)
            new_state, reward, done, info = self.env.step(action)
            state = new_state.reshape([1, -1])
        self.env.render(close=True)
        
    def plot(self, mode=None):
        if (mode == 'ci'):
            mean = np.mean(self.tracker, axis=1)
            std  = np.std(self.tracker, axis=1)
            Z=1.96
            ci95 = Z*std/np.sqrt(self.runs)
            plt.plot(self.X, mean, color='blue')
            plt.fill_between(self.X, mean-ci95, mean+ci95, color='blue', alpha=0.2)
            plt.title("Mean rewards over episodes +95% CI")

        else:
            plt.plot(self.tracker)
            plt.title("Rewards over Episodes")
        plt.show()
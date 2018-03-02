import numpy as np
from util import *
from base_agent import BaseAgent
from keras import layers, models, optimizers
from keras import backend as K
from keras.regularizers import *
from replay import ReplayBuffer

class DQN_Model():
    def __init__(self, edim, adim, learning_rate):
        self.edim = edim
        self.adim = adim
        
         # Input is environment vector
        inp = layers.Input(shape=(edim,), name='Input')
        
        net = layers.Dense(10, activation=None)(inp)
#         net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(10, activation=None)(net)
        
#         net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        
        actions = layers.Dense(adim, activation=None)(net)
        
        self.model = models.Model(inputs=inp, outputs=actions)
#         self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate, clipnorm=0.5))
        
        choice = layers.Input(batch_shape=(None,), name='Choice', dtype='int32')
        target = layers.Input(batch_shape=(None,), name='Target')
        
        # Manual Train function
        self.Qs = K.function(inputs=[inp], outputs=[actions])
        
        mult = K.one_hot(choice, self.adim)*actions
        self.mult = K.function(inputs=[inp, choice], outputs=[mult])
        
        chosen = K.sum(K.one_hot(choice, self.adim)*actions, axis=1)
        self.chosen = K.function(inputs=[inp, choice], outputs=[chosen])
        
        td = target-K.sum(K.one_hot(choice, self.adim)*actions, axis=1)
        self.td = K.function(inputs=[inp, choice, target], outputs=[td])
        
        loss = K.mean(K.square(target-K.sum(K.one_hot(choice, self.adim)*actions, axis=1)))
        self.loss = K.function(inputs=[inp, choice, target], outputs=[loss])
        
        optimizer = optimizers.Adam(lr=learning_rate, clipnorm=0.5)
        updates = optimizer.get_updates(loss, self.model.trainable_weights)
        self.train = K.function(inputs=[inp, choice, target], outputs=[], updates=updates)
        

class DQN(BaseAgent):
    def __init__(self, env):
        self.buffer_size = 20000
        self.batch_size = 64
        self.tau = 1
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        # Exploration Parameters
        self.E_start = 1
        self.E_end   = 0.1
        self.E_decay = 0.002
        self.episode = 0
        
        self.env = env
        self.os = self.env.observation_space
#         self.acs = self.env.action_space
        
        self.edim = len(self.os.high)
        self.adim = self.env.action_space.n
        
        self.buffer = ReplayBuffer(self.buffer_size, self.edim, 1)
        
        self.local  = DQN_Model(self.edim, self.adim, self.learning_rate)
        self.target = DQN_Model(self.edim, self.adim, self.learning_rate)
        
        self.initial_weights = self.local.model.get_weights()
        self.target.model.set_weights(self.initial_weights)
        
    def act(self, state, testing):        
        state = state.reshape([1, -1])
        actionQs = self.local.model.predict(state)
        
        action = np.argmax(actionQs)
        
        epsilon = self.E_end + (self.E_start - self.E_end)*np.exp(-self.E_decay*self.episode)
        if (not testing):
            if (np.random.rand() < epsilon):
                action = np.random.choice(self.adim)
        
#         action = np.array([action])
        
        return action
    
    def learn(self, state, action, reward, next_state, done, testing):
        
        # Skip all learning during testing
        if (testing):
            return
        
        act_index = action
        
        self.buffer.add(state, act_index, reward, next_state, done)
        
        if (done):
            self.episode += 1
        
        # TODO When upgrading to RDPG this should be per-episode based  

            states, actions, rewards, next_states, dones = self.buffer.batch(self.batch_size)

            actions.astype(int)
            actions = actions.reshape([-1])
            rewards = rewards.reshape([-1])
            dones = dones.reshape([-1])
            # Bellman equation
            target_Q = rewards + self.gamma*np.amax(self.target.model.predict_on_batch(next_states), axis=1)*(1-dones)

            self.local.train([states, actions, target_Q])

            self.soft_update(self.target, self.local)
        
        
    def soft_update(self, target, local):
        local_weights = np.array(local.model.get_weights())
        target_weights = np.array(target.model.get_weights())
        
        new_target_weights = (1-self.tau)*target_weights + self.tau*local_weights
        
        target.model.set_weights(new_target_weights)
        
    def reset(self):
        shuffle_weights(self.local.model, self.initial_weights)
        self.target.model.set_weights(self.local.model.get_weights())
        self.episode = 0
        
    
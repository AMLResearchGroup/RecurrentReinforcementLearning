from collections import deque
import random
import numpy as np

class ReplayBuffer():

    num_state = 3
    
    def __init__(self, buffer_size, edim, adim):
        " Initializes the replay buffer by creating a deque() and setting the size and buffer count. "
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.count = 0
        # for automatically reshaping the batches
        self.edim = edim
        self.adim = adim
         
    def add(self, s, a, r, s2, d):
         
        """ Adds new experience to the ReplayBuffer(). If the buffer size is
        reached, the oldest item is removed.
         
        Inputs needed to create new experience:
            s      - State
            a      - Action
            r      - Reward
            d      - Done
            s2     - Resulting State     
        """
        d = 1 if d else 0
        # Create experience list
        experience = (s, a, r, s2, d)
        
        # Check the size of the buffer
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
            
        # Add experience to buffer
        self.buffer.append(experience)
        
    def size(self):
        " Return the amount of stored experiences. " 
        return self.count
    
    def batch(self, batch_size):
        "Return a \"batch_size\" number of random samples from the buffer."
        
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
            batch_size = self.count
        else:
            batch = random.sample(self.buffer, batch_size)
            
        batch_state = np.array([item[0] for item in batch]).reshape([batch_size, self.edim])
        batch_action = np.array([item[1] for item in batch]).reshape([batch_size, self.adim])
        batch_reward = np.array([item[2] for item in batch]).reshape([batch_size, 1])
        batch_next_state = np.array([item[3] for item in batch]).reshape([batch_size,self.edim])
        batch_done = np.array([item[4] for item in batch]).reshape([batch_size, 1])
        
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done
            
    def clear(self):
        " Remove all entries from the ReplayBuffer. "
        self.buffer.clear()
        self.count = 0
        
        

class DumbReplayBuffer():
    
    def __init__(self, buffer_size, edim, adim):
        " Initializes the replay buffer by creating a deque() and setting the size and buffer count. "
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.count = 0
        # for automatically reshaping the batches
        self.edim = edim
        self.adim = adim
         
    def add(self, s, a, r, s2, d):
         
        """ Adds new experience to the ReplayBuffer(). If the buffer size is
        reached, the oldest item is removed.
         
        Inputs needed to create new experience:
            s      - State
            a      - Action
            r      - Reward
            d      - Done
            s2     - Resulting State     
        """
        d = 1 if d else 0
        # Create experience list
        experience = (s, a, r, s2, d)
        
        # Check the size of the buffer
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
            
        # Add experience to buffer
        self.buffer.append(experience)
        
    def size(self):
        " Return the amount of stored experiences. " 
        return self.count
    
    def batch(self, batch_size, seq_len):
        "Return a \"batch_size\" number of random samples from the buffer."
        
        if self.count <= seq_len :
            return (None, None, None, None, None)
        
        batch_state = np.zeros([batch_size, seq_len, self.edim])
        batch_action = np.zeros([batch_size, seq_len, self.adim])
        batch_reward = np.zeros([batch_size, seq_len])
        batch_next_state = np.zeros([batch_size, seq_len, self.edim])
        batch_done = np.zeros([batch_size, seq_len])
            
        
        batch_indices = np.random.choice(range(self.count-seq_len), batch_size)
        for i in range(batch_size):
            for j in range(seq_len):
                exp = self.buffer[batch_indices[i]+j]
                batch_state[i,j,:] = exp[0]
                batch_action[i,j,:] = exp[1]
                batch_reward[i,j] = exp[2]
                batch_next_state[i,j,:] = exp[3]
                batch_done[i,j] = exp[4]
        
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done
            
    def clear(self):
        " Remove all entries from the ReplayBuffer. "
        self.buffer.clear()
        self.count = 0
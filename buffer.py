import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque

Buffer_Size = int(1e5)  # replay buffer size
Batch_Size = 64         # minibatch size
gamma = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Replay_Buffer:
    
    def __init__(self, action_size, Buffer_Size, Batch_Size,seed):
             
        self.action_size = action_size
        self.memory = deque(maxlen=Buffer_Size)  
        self.Batch_Size = Batch_Size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state,action,reward,next_state,done):
        
        e = self.experience(state,action,reward,next_state,done) 
        # add state,action... values to the named tuple self.experience
        
        return self.memory.append(e)
    
    def sample(self):
        
        experiences = random.sample(self.memory, k = self.Batch_Size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)    
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)  
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)        
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        return len(self.memory)
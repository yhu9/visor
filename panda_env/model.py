import itertools
import math
import random
import os
import sys
from collections import namedtuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import matplotlib.pyplot as plt
from torch.autograd import Variable

from logger import Logger

##########################################################
#POSSIBLY ADD THIS LATER TO STABILIZE TRAINING
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity,device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device=device

    def push(self, state,action,next_state,reward,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        state = torch.Tensor(state)
        action = torch.Tensor(action).long()
        next_state = torch.Tensor(next_state)
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])
        self.memory[self.position] = Transition(state,action,next_state,reward,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        s = [None] * batch_size
        a = [None] * batch_size
        r = [None] * batch_size
        s2 = [None] * batch_size
        d = [None] * batch_size
        for i,e in enumerate(experiences):
            s[i] = e.state
            a[i] = e.action
            r[i] = e.reward
            s2[i] = e.next_state
            d[i] = e.done

        states = torch.stack(s).to(self.device).permute(0,3,1,2)
        actions = torch.stack(a).to(self.device)
        rewards = torch.stack(r).to(self.device)
        next_states = torch.stack(s2).to(self.device).permute(0,3,1,2)
        dones = torch.stack(d).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

#ACTION NET WITH LIMITED ACTION SPACE
class DQN1(nn.Module):

    def __init__(self):
        super(DQN1,self).__init__()

        self.res18 = resnet.resnet18()
        self.m1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,5)
                )
        self.m2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,5)
                )
        self.m3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,3)
                )

    def forward(self,frame):
        x = self.res18(frame)
        out1 = self.m1(x)
        out2 = self.m2(x)
        out3 = self.m3(x)
        return out1,out2,out3

##########################################################
#BASED ON PYTORCH EXAMPLE
#ACTION NET WITH LIMITED ACTION SPACE
class DQN(nn.Module):

    def __init__(self):
        super(DQN,self).__init__()

        self.res18 = resnet.resnet18()
        self.m1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,35)
                )
        self.m2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,15)
                )
        self.m3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,35)
                )
        self.m4 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,15)
                )
        self.m5 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,10)
                )

    def forward(self,frame):
        x = self.res18(frame)
        out1 = self.m1(x)
        out2 = self.m2(x)
        out3 = self.m3(x)
        out4 = self.m4(x)
        out5 = self.m5(x)
        return out1,out2,out3,out4,out5

#OUR MAIN MODEL WHERE ALL THINGS ARE RUN
class Model():
    def __init__(self,load=False):

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)

        #LOGGER FOR VISUALIZATION
        #self.logger = Logger('./logs')

        #DEFINE ALL NETWORK PARAMS
        self.EPISODES = 0
        self.BATCH_SIZE = 10
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 400
        self.TARGET_UPDATE = 10
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 1
        self.memory = ReplayMemory(10000,device=self.device)

        #OUR NETWORK
        self.model= DQN1()
        self.target_net = DQN1()
        self.model.to(self.device)
        self.target_net.to(self.device)

        #LOAD THE MODULES
        if load:
            self.model.load_state_dict(torch.load(load));
            self.target_net.load_state_dict(torch.load(load));
        else:
            self.model.apply(init_weights)
            self.target_net.apply(init_weights)


        #DEFINE OPTIMIZER AND HELPER FUNCTIONS
        self.opt = torch.optim.Adam(itertools.chain(self.model.parameters()),lr=0.00001,betas=(0.0,0.9))
        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

    def load(self,model_dir):
        self.model.load_state_dict(torch.load(model_dir))
        self.target_net.load_state_dict(torch.load(model_dir))

    def optimize(self):
        #don't bother if you don't have enough in memory
        if len(self.memory) < self.BATCH_SIZE: return 0.0


        self.model.train()
        s1,actions,r1,s2,d = self.memory.sample(self.BATCH_SIZE)

        #get old Q values and new Q values for belmont eq
        qvals = self.model(s1)


        q1 = qvals[0].gather(1,actions[:,0].unsqueeze(1))
        q2 = qvals[1].gather(1,actions[:,1].unsqueeze(1))
        q3 = qvals[2].gather(1,actions[:,2].unsqueeze(1))
        state_action_values = torch.cat((q1,q2,q3),-1)

        with torch.no_grad():
            qvals_t = self.target_net(s2)
            q1_t = qvals_t[0].max(1)[0].unsqueeze(1)
            q2_t = qvals_t[1].max(1)[0].unsqueeze(1)
            q3_t = qvals_t[2].max(1)[0].unsqueeze(1)
            q_target = torch.cat((q1_t,q2_t,q3_t),-1)

        expected_state_action_values = (q_target * self.GAMMA) * (1-d) + r1

        #LOSS IS l2 loss of belmont equation
        loss = self.l2(state_action_values,expected_state_action_values)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    #GREEDY ACTION SELECTION
    def select_greedy(self,state):
        with torch.no_grad():
            data = torch.from_numpy(np.ascontiguousarray(state)).float().to(self.device)
            data = data.permute(2,0,1).unsqueeze(0)
            a1,a2,a3 = self.model(data)
            return a1.max(1)[1].item(), a2.max(1)[1].item(),a3.max(1)[1].item()

    #STOCHASTIC ACTION SELECTION WITH DECAY TOWARDS GREEDY SELECTION. Actions are represented as onehot values
    def select_action(self,state):
        with torch.no_grad():
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
            self.steps += 1
            if sample > eps_threshold:
                data = torch.from_numpy(np.ascontiguousarray(state)).float().to(self.device)
                data = data.permute(2,0,1).unsqueeze(0)
                #send the state through the DQN and get the index with the highest value for that state
                a1,a2,a3 = self.model(data)
                return a1.max(1)[1].item(), a2.max(1)[1].item(),a3.max(1)[1].item()
            else:
                return random.randrange(5), random.randrange(5),random.randrange(3)

    #FORWARD PASS
    def forward(self,rgb):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        x = x.permute(2,0,1)
        x = x.unsqueeze(0)
        actions = self.net(x)

        return actions

    def save(self):
        if not os.path.isdir('model'): os.mkdir('model')
        torch.save(self.model.state_dict(),'model/DQN_1_6.pth')

if __name__ == '__main__':
    #net = models.resnet14(pretrained=False,channels=6)
    #print(net.forward(torch.zeros((1,3,224,224))).shape)
    print('hello world')




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

    def push(self,state,action,next_state,reward,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        v,s = state
        v2,s2 = next_state
        v = torch.Tensor(v)
        s = torch.Tensor(s)
        state = (v,s)
        v2 = torch.Tensor(v2)
        s2 = torch.Tensor(s2)
        next_state = (v2,s2)
        action = torch.Tensor(action).long()
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])
        self.memory[self.position] = Transition(state,action,next_state,reward,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        #experiences = self.memory[-batch_size:]
        v = [None] * batch_size
        s = [None] * batch_size
        a = [None] * batch_size
        r = [None] * batch_size
        v2 = [None] * batch_size
        s2 = [None] * batch_size
        d = [None] * batch_size
        for i,e in enumerate(experiences):
            v[i],s[i] = e.state
            a[i] = e.action
            r[i] = e.reward
            v2[i],s2[i] = e.next_state
            d[i] = e.done

        v = torch.stack(v).to(self.device)
        states = torch.stack(s).to(self.device).permute(0,3,1,2)
        actions = torch.stack(a).to(self.device)
        rewards = torch.stack(r).to(self.device)
        v2 = torch.stack(v2).to(self.device)
        next_states = torch.stack(s2).to(self.device).permute(0,3,1,2)
        dones = torch.stack(d).to(self.device)
        return (v,states), actions, rewards, (v2,next_states), dones

    def __len__(self):
        return len(self.memory)

#DDQN network
class DDQN(nn.Module):
    def __init__(self):
        super(DDQN,self).__init__()

        self.res18 = resnet.resnet18()
        self.h1 = nn.Linear(1005,256)

        self.value = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,1)
                )

        self.action = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,243)
                )

    def forward(self,state):
        v,frame = state
        x = self.res18(frame)
        x = torch.cat((v,x),dim=-1)
        x = self.h1(x)
        v = self.value(x)
        a = self.action(x)
        q = v + (a - torch.mean(a))
        return q

#DDPG with limited action space
class DQN(nn.Module):

    def __init__(self):
        super(DQN,self).__init__()

        self.res18 = resnet.resnet18()
        self.action = nn.Sequential(
                nn.Linear(1000+5,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256,243)
                )

    def forward(self,state):
        v,frame = state
        x = self.res18(frame)
        x = torch.cat((v,x),dim=-1)
        a = self.action(x)
        return a

#OUR MAIN MODEL WHERE ALL THINGS ARE RUN
class Model():
    def __init__(self,load=False,mode='DQN'):

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #DEFINE ALL NETWORK PARAMS
        self.EPISODES = 0
        self.BATCH_SIZE = 32
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.TARGET_UPDATE = 20
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 1
        self.memory = ReplayMemory(10000,device=self.device)

        print(self.device)
        #OUR NETWORK
        if mode == 'DQN':
            self.model= DQN()
            self.target_net = DQN()
        elif mode == 'DDQN':
            self.model= DDQN()
            self.target_net = DDQN()
        self.model.to(self.device)
        self.target_net.to(self.device)

        #LOAD THE MODULES
        if load:
            print('MODEL' + load + ' LOADED')
            self.model.load_state_dict(torch.load(load));
            self.target_net.load_state_dict(torch.load(load));
        else:
            self.model.apply(init_weights)
            self.target_net.apply(init_weights)

        #DEFINE OPTIMIZER AND HELPER FUNCTIONS
        self.opt = torch.optim.Adam(itertools.chain(self.model.parameters()),lr=0.0001,betas=(0.0,0.9))
        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

    def optimize(self):
        #don't bother if you don't have enough in memory
        if len(self.memory) < self.BATCH_SIZE: return 0.0

        self.model.train()
        s1,actions,r1,s2,d = self.memory.sample(self.BATCH_SIZE)
        a = actions[:,0] * 81 + actions[:,1] * 27 + actions[:,2] * 9 + actions[:,1] * 3 + actions[:,0] * 1
        a = a.unsqueeze(1)

        #get old Q values and new Q values for belmont eq
        qvals = self.model(s1)
        qvals = qvals.gather(1,a)

        #q1 = qvals[0].gather(1,actions[:,0].unsqueeze(1))
        #q2 = qvals[1].gather(1,actions[:,1].unsqueeze(1))
        #q3 = qvals[2].gather(1,actions[:,2].unsqueeze(1))
        #state_action_values = torch.cat((q1,q2,q3),-1)

        with torch.no_grad():
            qvals_t = self.target_net(s2)
            qvals_t = qvals_t.max(1)[0].unsqueeze(1)
            #q1_t = qvals_t[0].max(1)[0].unsqueeze(1)
            #q2_t = qvals_t[1].max(1)[0].unsqueeze(1)
            #q3_t = qvals_t[2].max(1)[0].unsqueeze(1)
            #q_target = torch.cat((q1_t,q2_t,q3_t),-1)

        expected_state_action_values = (qvals_t * self.GAMMA) * (1-d) + r1

        #print(r1)
        #print(expected_state_action_values)

        #LOSS IS l2 loss of belmont equation
        #loss = self.l2(state_action_values,expected_state_action_values)
        loss = self.l2(qvals,expected_state_action_values)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    #GREEDY ACTION SELECTION
    def select_greedy(self,state):
        self.model.eval()
        with torch.no_grad():
            v,s = state
            frame = torch.from_numpy(np.ascontiguousarray(s)).float().to(self.device)
            frame = frame.permute(2,0,1).unsqueeze(0)
            v = torch.Tensor(v).unsqueeze(0).to(self.device)
            data = (v,frame)
            a = self.model(data)
            idx = a.max(1)[1].item()
            a1,a2,a3,a4,a5 = np.where(np.arange(243).reshape((3,3,3,3,3)) == idx)
            return a1[0], a2[0],a3[0],a4[0],a5[0]

    #STOCHASTIC ACTION SELECTION WITH DECAY TOWARDS GREEDY SELECTION. Actions are represented as onehot values
    def select_action(self,state):
        self.model.eval()
        with torch.no_grad():
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
            self.steps += 1
            if sample > eps_threshold:
                frame = torch.from_numpy(np.ascontiguousarray(state[1])).float().to(self.device)
                frame = frame.permute(2,0,1).unsqueeze(0)
                v = torch.Tensor(state[0]).unsqueeze(0).to(self.device)
                #send the state through the DQN and get the index with the highest value for that state
                a = self.model((v,frame))
                idx = a.max(1)[1].item()
                a1,a2,a3,a4,a5 = np.where(np.arange(243).reshape((3,3,3,3,3)) == idx)
                return a1[0],a2[0],a3[0],a4[0],a5[0]
                #return a1.max(1)[1].item(), a2.max(1)[1].item(),a3.max(1)[1].item()
            else:
                return random.randrange(3), random.randrange(3), random.randrange(3), random.randrange(3),random.randrange(3)

    def save(self,outfile):
        if not os.path.isdir('model'): os.mkdir('model')
        torch.save(self.model.state_dict(),os.path.join('model',outfile))





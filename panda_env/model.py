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

from utils import ShadowDetector
from utils import LM_Detector
from logger import Logger

####################################################3

#RESIDUAL BLOCKS
class ResBlock(nn.Module):
    def __init__(self, in_features, out_features,k=3):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features,out_features,k,stride=1,padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features,out_features,k,stride=1,padding=1)
        )

        if in_features == out_features:
            self.skip = None
        else:
            self.skip = nn.Conv2d(in_features,out_features,1,1,0)

    def forward(self, x):
        if not self.skip:
            x = self.block(x)
        else:
            x = self.block(x) + self.skip(x)
        return x

##########################################################
#POSSIBLY ADD THIS LATER TO STABILIZE TRAINING
Transition = namedtuple('Transition',('state', 'action1','action2','action3', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
##########################################################

#ACTION NET WITH LIMITED ACTION SPACE
class DQN(nn.Module):

    def __init__(self):
        super(DQN,self).__init__()

        #self.res18 = models.resnet18()
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

    def forward(self,x):
        x = self.res18(x)
        out1 = self.m1(x)
        out2 = self.m2(x)
        out3 = self.m3(x)
        return out1,out2,out3

#ACTION NET WITH FULL ACTION SPACE
class anet2(nn.Module):

    def __init__(self,channel=3):
        super(anet2,self).__init__()

        self.m1 = nn.Sequential(
                nn.Conv2d(channel,32,3,stride=1,padding=1),
                ResBlock(32,32),
                ResBlock(32,64),
                ResBlock(64,64),
                ResBlock(64,1),
                )

    def forward(self,x):
        out = self.m1(x)
        return out


#OUR MAIN MODEL WHERE ALL THINGS ARE RUN
class Model():
    def __init__(self,load=False):

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)

        #LOGGER FOR VISUALIZATION
        self.logger = Logger('./logs')

        #DEFINE ALL NETWORK PARAMS
        self.EPISODES = 0
        self.BATCH_SIZE = 10
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 400
        self.TARGET_UPDATE = 5
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 1
        self.memory = ReplayMemory(10000)

        #EVALUATOR
        self.shadowfunc = ShadowDetector()
        self.lmfunc = LM_Detector()

        #OUR NETWORK
        self.pnet = DQN()
        self.vnet = DQN()

        #LOAD THE MODULES
        if load:
            self.pnet.load_state_dict(torch.load(load));
            self.vnet.load_state_dict(torch.load(load));
        else:
            self.pnet.apply(init_weights)
            self.vnet.apply(init_weights)

        #DEFINE OPTIMIZER AND HELPER FUNCTIONS
        self.opt = torch.optim.Adam(itertools.chain(self.pnet.parameters()),lr=0.00001,betas=(0.0,0.9))
        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

    #OPTIMIZE THE NETWORK BY SAMPLING FROM THE REPLAY BUFFER
    def optimize(self):

        if len(self.memory) < self.BATCH_SIZE: return 0.0

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=self.device,dtype=torch.uint8)
        next_states = [s for s in batch.next_state if s is not None]

        state_batch = torch.cat(batch.state)
        action_batch1 = torch.cat(batch.action1)
        action_batch2 = torch.cat(batch.action2)
        action_batch3 = torch.cat(batch.action3)
        reward_batch = torch.cat(batch.reward)

        #get old Q values and new Q values for belmont eq
        a1,a2,a3 = self.pnet(state_batch)
        state_action_values1 = a1.gather(1,action_batch1)
        state_action_values2 = a2.gather(1,action_batch2)
        state_action_values3 = a3.gather(1,action_batch3)
        if len(next_states) == 0:
            expected_state_action_values1 = reward_batch
            expected_state_action_values2 = reward_batch
            expected_state_action_values3 = reward_batch
        else:
            non_final_next_states = torch.cat(next_states)
            nextstate_values1 = torch.zeros(self.BATCH_SIZE,device=self.device)
            nextstate_values2 = torch.zeros(self.BATCH_SIZE,device=self.device)
            nextstate_values3 = torch.zeros(self.BATCH_SIZE,device=self.device)
            v1,v2,v3 = self.vnet(non_final_next_states)
            nextstate_values1[non_final_mask] = v1.max(1)[0].detach()
            nextstate_values2[non_final_mask] = v2.max(1)[0].detach()
            nextstate_values3[non_final_mask] = v3.max(1)[0].detach()
            expected_state_action_values1 = (nextstate_values1 * self.GAMMA) + reward_batch
            expected_state_action_values2 = (nextstate_values2 * self.GAMMA) + reward_batch
            expected_state_action_values3 = (nextstate_values3 * self.GAMMA) + reward_batch

        #LOSS IS l2 loss of belmont equation
        loss = self.l2(state_action_values1,expected_state_action_values1.unsqueeze(1)) + self.l2(state_action_values2,expected_state_action_values2.unsqueeze(1)) + self.l2(state_action_values3,expected_state_action_values3.unsqueeze(1))

        self.opt.zero_grad()
        loss.backward()
        for param in self.pnet.parameters():
            param.grad.data.clamp_(-1,1)
        self.opt.step()

        return loss


    #GREEDY ACTION SELECTION
    def select_greedy(self,state):
        with torch.no_grad():
            a1,a2,a3 = self.pnet(state)
            return a1.max(1)[1].view(1,1), a2.max(1)[1].view(1,1),a3.max(1)[1].view(1,1)

    #STOCHASTIC ACTION SELECTION WITH DECAY TOWARDS GREEDY SELECTION. Actions are represented as onehot values
    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
        self.steps += 1

        if sample > eps_threshold:
            with torch.no_grad():
                #send the state through the DQN and get the index with the highest value for that state
                a1,a2,a3 = self.pnet(state)

                return a1.max(1)[1].view(1,1), a2.max(1)[1].view(1,1),a3.max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(5)]],device=self.device,dtype=torch.long), torch.tensor([[random.randrange(5)]],device=self.device,dtype=torch.long),torch.tensor([[random.randrange(3)]],device=self.device,dtype=torch.long)

    #FORWARD PASS
    def forward(self,rgb):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        x = x.permute(2,0,1)
        x = x.unsqueeze(0)
        actions = self.net(x)

        return actions

    #REWARD MAP GENERATION GIVEN STATE S
    def genReward(self,params,rgb,lm,draw=False):
        rgb = (rgb * 255).astype(np.uint8)
        eye_mask = self.lmfunc.get_eyes(rgb,lm)     #((cx,cy),(h,w),rotation)
        #self.lmfunc.view_lm(rgb,lm)                #for visualization
        shadow = self.shadowfunc.get_shadow(rgb)

        IOU = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(np.logical_or(eye_mask,shadow))
        EYE = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(eye_mask)

        #525 = 15 * 35 which is the area of the visor roughly
        #params are the visor parameters (x,y,w,h,theta)
        A = (params[2] * params[3]) / np.sum(eye_mask)
        threshold = EYE + IOU

        #reward
        if threshold > 1.2:
            reward,flag = torch.Tensor([threshold - A]), True
        else:
            reward,flag = torch.Tensor([-0.20]), False

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(params,eye_mask,shadow,rgb.copy(),lm,IOU,EYE,A,reward)

        return reward,flag

    def drawReward(self,params,eye_mask,shadow,rgb,lm,IOU,EYE,A,reward):

        h,w,d = rgb.shape
        img = np.zeros((h,w+100,d))

        rgb[eye_mask] = rgb[eye_mask] * [0,0,1]
        rgb[shadow] = rgb[shadow] * [1,0,0]

        A = (params[2] * params[3]) / 300.0
        IOU = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(np.logical_or(eye_mask,shadow))
        EYE = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(eye_mask)

        #show landmarks
        for x,y in lm:
            cv2.circle(rgb,(x,y),2,(255,0,255),-1)

        #IMG IS 224 X 112
        cv2.putText(img,"IOU %.3f" % IOU,(2,150),2,0.4,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,"EYE %.3f" % EYE,(2,170),2,0.4,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,"A %.3f" % A,(2,190),2,0.4,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,"reward %.3f" % reward ,(2,210),2,0.4,(0,255,0),1,cv2.LINE_AA)

        img[:,100:,:] = rgb
        cv2.imshow('semantic mask',img)
        cv2.waitKey(10)

    def save(self):
        if not os.path.isdir('model'): os.mkdir('model')
        torch.save(self.pnet.state_dict(),'model/DQN.pth')

if __name__ == '__main__':
    #net = models.resnet14(pretrained=False,channels=6)
    #print(net.forward(torch.zeros((1,3,224,224))).shape)
    print('hello world')




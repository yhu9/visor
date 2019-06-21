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
from torch.autograd import Variable
#from torch.distributions import Categorical
from torch.distributions.normal import Normal
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
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

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

    def reset(self):
        del self.memory[:]
        self.position = 0

    def sample(self, batch_size):

        transitions = random.sample(self.memory,batch_size)
        batch = Transition(*zip(*transitions))
        next_states = [s for s in batch.next_state]
        curr_states = [s for s,v in zip(batch.state,batch.next_state)]

        s1 = torch.cat(curr_states)
        actions = np.concatenate(batch.action,axis=0)
        a1 = Variable(torch.from_numpy(actions))
        r1 = torch.cat(batch.reward).unsqueeze(1)
        s2 = torch.cat(next_states)

        return s1,a1, r1, s2

    def __len__(self):
        return len(self.memory)

##########################################################
#DDPG
#https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/model.py
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):

        def __init__(self, action_dim=5):
                """
                :param state_dim: Dimension of input state (int)
                :param action_dim: Dimension of input action (int)
                :return:
                """
                EPS = 0.003
                super(Critic, self).__init__()

                self.action_dim = action_dim

                self.res18 = resnet.resnet18()
                self.fcs = nn.Linear(1000,128)
                self.fca = nn.Linear(action_dim,128)
                self.fc = nn.Linear(256,128)
                self.out = nn.Linear(128,1)

        def forward(self, state, action):
                """
                returns Value function Q(s,a) obtained from critic network
                :param state: Input state (Torch Variable : [n,state_dim] )
                :param action: Input Action (Torch Variable : [n,action_dim] )
                :return: Value function : Q(S,a) (Torch Variable : [n,1] )
                """

                s = F.relu(self.res18(state))
                s = F.relu(self.fcs(s))
                a = F.relu(self.fca(action))
                x = torch.cat((s,a),dim=1)
                x = F.relu(self.fc(x))
                x = self.out(x)

                return x

class Actor(nn.Module):
        def __init__(self, action_dim=5, action_lim=5.0):
                """
                :param state_dim: Dimension of input state (int)
                :param action_dim: Dimension of output action (int)
                :param action_lim: Used to limit action in [-action_lim,action_lim]
                :return:
                """
                super(Actor, self).__init__()

                self.action_dim = action_dim
                self.action_lim = action_lim

                self.res18 = resnet.resnet18()
                self.out = nn.Linear(1000,action_dim)

        def forward(self, state):
                """
                returns policy function Pi(s) obtained from actor network
                this function is a gaussian prob distribution for all actions
                with mean lying in (-1,1) and sigma lying in (0,1)
                The sampled action can , then later be rescaled
                :param state: Input state (Torch Variable : [n,state_dim] )
                :return: Output action (Torch Variable: [n,action_dim] )
                """

                x = F.relu(self.res18(state))
                action = torch.tanh(self.out(x))
                action = action * self.action_lim

                return action

##########################################################

#OUR MAIN MODEL WHERE ALL THINGS ARE RUN
class Model():
    def __init__(self,load=False):

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #LOGGER FOR VISUALIZATION
        self.logger = Logger('./logs')

        #DEFINE ALL NETWORK PARAMS
        self.EPISODES = 0
        self.BATCH_SIZE = 10
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 400
        self.GAMMA = 0.999
        self.LR = 0.00001
        self.TARGET_UPDATE = 5
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 1
        self.memory = ReplayMemory(10000)

        #EVALUATOR
        self.shadowfunc = ShadowDetector()
        self.lmfunc = LM_Detector()

        #OUR DDPG NETWORK
        self.actor = Actor()
        self.target_actor = Actor()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),self.LR)
        self.critic = Critic()
        self.target_critic = Critic()
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),self.LR)

        #LOAD THE MODULES
        if False:   #fix this later
            self.actor.load_state_dict(torch.load(load));
            self.critic.load_state_dict(torch.load(load));
        else:
            self.actor.apply(init_weights)
            self.target_actor.apply(init_weights)
            self.critic.apply(init_weights)
            self.target_critic.apply(init_weights)

        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

    #OPTIMIZE THE NETWORK BY SAMPLING FROM THE REPLAY BUFFER
    def optimize(self):

        #R = 0
        #eps = np.finfo(np.float32).eps.item()
        if len(self.memory) < self.BATCH_SIZE: return 0

        #get values from memory buffer
        s1,a1,r1,s2 = self.memory.sample(self.BATCH_SIZE)

        #optmize critic
        a2 = self.target_actor(s2).detach()
        next_val = self.target_critic.forward(s2,a2).detach().squeeze(0)
        y_expected = r1 + self.GAMMA * next_val     #Q expected
        y_predicted = self.critic.forward(s1,a2)    #Q predicted
        loss_critic = self.l1(y_predicted,y_expected)
        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

        #optimize actor
        pred_a1 = self.actor(s1)
        loss_actor = -1 * torch.sum(self.critic(s1,pred_a1))
        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        print(loss_actor,loss_critic)

        return loss_actor + loss_critic

    #STOCHASTIC ACTION SELECTION WITH DECAY TOWARDS GREEDY SELECTION. Actions are represented as onehot values
    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
        self.steps += 1
        with torch.no_grad():
            action = self.actor(state)
            if sample > eps_threshold:
                return action.data.numpy()
            else:
                return action.data.numpy() + np.random.normal(0,2,(1,5))

    #GREEDY ACTION SELECTION
    def select_greedy(self,state):
        with torch.no_grad():
            a1,a2,a3 = self.pnet(state)
            return a1.max(1)[1].view(1,1), a2.max(1)[1].view(1,1),a3.max(1)[1].view(1,1)

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
        #SHADOW = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(shadow)

        #525 = 15 * 35 which is the area of the visor roughly
        #params are the visor parameters (x,y,w,h,theta)
        A = (params[2] * params[3]) / np.sum(eye_mask)
        threshold = EYE + IOU

        #reward
        if threshold > 1.2:
            reward,flag = torch.Tensor([threshold]), True
        else:
            reward,flag = torch.Tensor([-0.10]), False

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
        torch.save(self.actor.state_dict(),'model/actor.pth')
        torch.save(self.critic.state_dict(),'model/critic.pth')

if __name__ == '__main__':
    #net = models.resnet14(pretrained=False,channels=6)
    #print(net.forward(torch.zeros((1,3,224,224))).shape)
    print('hello world')




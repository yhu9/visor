import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.residual = resnet.resnet18()
        self.fc_block = torch.nn.Sequential(
                nn.Linear(1000+5,128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                nn.Linear(128,action_size),
                torch.nn.Tanh()
                )

    def reset_parameters(self):
        self.fc_block.apply(init_weights)

    def forward(self, frame,visor):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.residual(frame)
        x = torch.cat((x,visor),1)
        return self.fc_block(x)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size, seed, fc1_units=1000, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.residual = resnet.resnet18()
        self.fc_block = torch.nn.Sequential(
                nn.Linear(1000+5+action_size,128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                nn.Linear(128,action_size)
                )
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_block.apply(init_weights)

    def forward(self, frame,visor, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.residual(frame)
        x = torch.cat((xs,visor,action),1)
        return self.fc_block(x)




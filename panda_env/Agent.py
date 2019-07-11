import numpy as np
import random
import copy
import math
from collections import namedtuple, deque

from model_mountaincar.Model import Actor, Critic
from model_mountaincar.Noise import OUNoise
from model import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
#EPSILON_MAX = 1.0
#EPSILON_MIN = 0.1
EPSILON_DECAY = 1e-3
LEARN_START = 10
UPDATE_EVERY = 1
UPDATES_PER_STEP = 1


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, random_seed,load=False):
        """Initialize an Agent object.
        Params
        ======
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.BATCH_SIZE = 10          # minibatch size
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.steps = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        #self.epsilon = EPSILON_MAX

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_size, random_seed).to(self.device)
        self.actor_target = Actor( action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic( action_size, random_seed).to(self.device)
        self.critic_target = Critic( action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        if load:
            idx = load.rfind('_')
            actor_path = load[:idx] + '_actor.pth'
            critic_path = load[:idx] +'_critic.pth'
            self.load(actor_path,critic_path)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, mu=0, theta=0.15, sigma=0.2)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE,device=self.device)

        # Make sure target is with the same weight as the source
        self.hard_update()
        self.t_step = 0

    def load(self,actor_name,critic_name):
        print('model loaded')
        self.actor_local.load_state_dict(torch.load(actor_name))
        self.critic_local.load_state_dict(torch.load(critic_name))
        self.hard_update()

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        sg1 = state.copy()
        sg1[:,:,-1] = next_state[:,:,-1]
        self.memory.push(sg1,action,reward - 1,next_state.copy(),done)
        if reward < 0.25: reward = -1
        else: reward += 1
        self.memory.push(state.copy(), action, reward, next_state.copy(), done)
        loss = 0.0,0.0

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                self.actor_local.train()
                s,a,r,s2,d = self.memory.sample(self.BATCH_SIZE)
                loss = self.learn((s,a,r,s2,d))
        return loss

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        #return random.randrange(5), random.randrange(5),random.randrange(3)
        self.actor_local.eval()
        with torch.no_grad():
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
            self.steps += 1
            if sample > eps_threshold or not add_noise:
                frame = torch.from_numpy(np.ascontiguousarray(state)).float()
                frame = frame.permute(2,0,1).unsqueeze(0).to(self.device)
                action = self.actor_local(frame)
                return action.cpu().data.numpy()
            else:
                return np.random.normal(0,0.5,(1,5))

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_pred = self.critic_local(states, actions)
        critic_loss = F.l1_loss(Q_pred, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_target(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ?_target = t*?_local + (1 - t)*?_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self):
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.actor_target.load_state_dict(self.actor_local.state_dict())

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity,device='cpu'):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = []
        self.device=device
        self.capacity = capacity
        self.Transition = namedtuple("Experience",field_names = ["state","action","reward","next_state","done"])
        self.seed = random.seed()
        self.position = 0

    def push(self, state,action,reward,next_state,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        state = torch.Tensor(state)
        action = torch.Tensor(action).long()
        next_state = torch.Tensor(next_state)
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])
        self.memory[self.position] = self.Transition(state,action,reward,next_state,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        frames = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(self.device)
        frames = frames.permute(0,3,1,2)
        states = frames
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_frames = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(self.device)
        next_frames = next_frames.permute(0,3,1,2)
        next_states = next_frames
        #next_vparams = torch.from_numpy(np.stack([e.next_state[1] for e in experiences if e is not None])).float().to(self.device)
        #next_states = next_frames,next_vparams
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)








import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_mountaincar.Model import Actor, Critic
from model_mountaincar.Noise import OUNoise
from model import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 10          # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 1e-3
LEARN_START = 10
UPDATE_EVERY = 1
UPDATES_PER_STEP = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, random_seed,load=False):
        """Initialize an Agent object.
        Params
        ======
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON_MAX

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_size, random_seed).to(device)
        self.actor_target = Actor( action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic( action_size, random_seed).to(device)
        self.critic_target = Critic( action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, mu=0, theta=0.15, sigma=0.2)

        # Replay memory
        self.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory2 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

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
        if reward > 0.0:
            self.memory1.add(state.copy(), action, reward, next_state.copy(), done)
        else:
            self.memory2.add(state.copy(), action, reward, next_state.copy(), done)
        loss = 0.0,0.0

        # Learn, if enough samples are available in memory
        if len(self.memory1) > BATCH_SIZE and len(self.memory2) > BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                good_experience = self.memory1.sample()
                bad_experience = self.memory2.sample()
                s = torch.cat((good_experience[0],bad_experience[0]),0)
                a = torch.cat((good_experience[1],bad_experience[1]),0)
                r = torch.cat((good_experience[2],bad_experience[2]),0)
                s2 = torch.cat((good_experience[3],bad_experience[3]),0)
                d = torch.cat((good_experience[4],bad_experience[4]),0)
                loss = self.learn((s,a,r,s2,d), GAMMA)
        return loss

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        frame = torch.from_numpy(np.ascontiguousarray(state)).float()
        frame = frame.permute(2,0,1).unsqueeze(0)
        #visor = torch.from_numpy(np.ascontiguousarray(visor)).float()
        #visor = visor.unsqueeze(0) / 35.0
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(frame).cpu().data.numpy()

        sample = random.random()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action = action + np.random.normal(0,self.epsilon * noise)

        return np.clip(action,-1,1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
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
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

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

        # ----------------------- update target networks ----------------------- #
        #self.soft_update(self.critic_local, self.critic_target, TAU)
        #self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        if self.epsilon - EPSILON_DECAY > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN

        self.noise.reset()

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

    def __init__(self, action_size, capacity, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.capacity = capacity
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.position = 0
        self.reset()

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state.copy(), action, reward, next_state.copy(), done)
        self.memory.append(e)

    def reset(self):
        self.memory = []

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        frames = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(device)
        frames = frames.permute(0,3,1,2)
        states = frames
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_frames = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(device)
        next_frames = next_frames.permute(0,3,1,2)
        next_states = next_frames
        #next_vparams = torch.from_numpy(np.stack([e.next_state[1] for e in experiences if e is not None])).float().to(device)
        #next_states = next_frames,next_vparams
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent2():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, random_seed):
        """Initialize an Agent object.
        Params
        ======
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON_MAX

        # or Network (w/ Target Network)
        self.dqn_local = DQN().to(device)
        self.dqn_target = DQN().to(device)
        self.dqn_optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, mu=0, theta=0.15, sigma=0.2)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Make sure target is with the same weight as the source
        self.hard_update()
        self.t_step = 0

    def load(self,dqn_name):
        self.dqn_local.load_state_dict(torch.load(dqn_name))

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        loss = 0.0

        if len(self.memory) > LEARN_START:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # Learn, if enough samples are available in memory
                if len(self.memory) > BATCH_SIZE:
                    for _ in range(UPDATES_PER_STEP):
                        experiences = self.memory.sample()
                        loss = self.learn(experiences, GAMMA)
        return loss

    def act(self, state):
        """Returns actions for given state as per current policy."""
        frame = torch.from_numpy(np.ascontiguousarray(state)).float()
        frame = frame.permute(2,0,1).unsqueeze(0)
        #visor = torch.from_numpy(np.ascontiguousarray(visor)).float()
        #visor = visor.unsqueeze(0) / 35.0

        sample = random.random()
        if sample < self.epsilon:
            return random.randint(0,34), random.randint(0,14),random.randint(0,34), random.randint(0,14),random.randint(0,9)

        self.dqn_local.eval()
        with torch.no_grad():
            a1,a2,a3,a4,a5 = self.dqn_local(frame)
        self.dqn_local.train()
        return a1.max(1)[1].item(), a2.max(1)[1].item(), a3.max(1)[1].item(),a4.max(1)[1].item(),a5.max(1)[1].item()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
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
        actions = actions.type(torch.LongTensor)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            qvals = self.dqn_target(next_states)
            Q_targets_next = torch.cat(([q.max(1)[0].unsqueeze(1) for q in qvals]),-1)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        qvals_local = self.dqn_local(states)
        #q1,q2,q3,q4,q5 = qvals_local
        q1 = qvals_local[0].gather(1,actions[:,0].unsqueeze(1))
        q2 = qvals_local[0].gather(1,actions[:,1].unsqueeze(1))
        q3 = qvals_local[0].gather(1,actions[:,2].unsqueeze(1))
        q4 = qvals_local[0].gather(1,actions[:,3].unsqueeze(1))
        q5 = qvals_local[0].gather(1,actions[:,4].unsqueeze(1))
        Q_pred = torch.cat((q1,q2,q3,q4,q5),-1)

        loss = F.mse_loss(Q_pred, Q_targets)

        # Minimize the loss
        self.dqn_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.dqn_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        #self.soft_update(self.dqn_local, self.dqn_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        if self.epsilon - EPSILON_DECAY > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN

        self.noise.reset()

        return loss.item()

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
        self.dqn_target.load_state_dict(self.dqn_local.state_dict())





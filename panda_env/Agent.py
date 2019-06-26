import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_mountaincar.Model import Actor, Critic
from model_mountaincar.Noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 5         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-2         # learning rate of the actor
LR_CRITIC = 5e-3        # learning rate of the critic
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
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Make sure target is with the same weight as the source
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        loss = 0.0,0.0

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

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        frame, visor = state
        frame = torch.from_numpy(np.ascontiguousarray(frame)).float()
        frame = frame.permute(2,0,1).unsqueeze(0)
        visor = torch.from_numpy(np.ascontiguousarray(visor)).float()
        visor = visor.unsqueeze(0) / 35.0
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(frame,visor).cpu().data.numpy()

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
        actions_next = self.actor_target(next_states[0],next_states[1])
        Q_targets_next = self.critic_target(next_states[0],next_states[1], actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_pred = self.critic_local(states[0],states[1], actions)
        critic_loss = F.mse_loss(Q_pred, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states[0],states[1])
        actor_loss = -self.critic_local(states[0],states[1], actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

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

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.reset()

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        frames = torch.from_numpy(np.stack([e.state[0] for e in experiences if e is not None])).float().to(device)
        frames = frames.permute(0,3,1,2)
        visor_params = torch.from_numpy(np.stack([e.state[1] for e in experiences if e is not None])).float().to(device)
        states = frames,visor_params

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_frames = torch.from_numpy(np.stack([e.next_state[0] for e in experiences if e is not None])).float().to(device)
        next_frames = next_frames.permute(0,3,1,2)
        next_vparams = torch.from_numpy(np.stack([e.next_state[1] for e in experiences if e is not None])).float().to(device)
        next_states = next_frames,next_vparams
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



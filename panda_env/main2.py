#!/usr/bin/env python

import time
import os
import numpy as np
from collections import deque
import argparse

import torch

#CUSTOM MODULES
from env import World

################################################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
env = World()
random_seed = 2

# size of each action
action_size = 5
print('Size of each action:', action_size)

from Agent import Agent
from logger import Logger

agent = Agent(action_size=action_size, random_seed=random_seed)
def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'model/checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'model/checkpoint_critic.pth')

def train(n_episodes=10000, max_t=10, print_every=1, save_every=10):
    logger = Logger('./logs')
    scores_deque = deque(maxlen=20)
    solved_deque = deque(maxlen=100)
    scores = []
    best = 0

    if opt.load:
        actor_path = os.path.join(opt.load,'checkpoint_actor.pth')
        critic_path = os.path.join(opt.load,'checkpoint_critic.pth')
        agent.load(actor_path,critic_path)

    for i_episode in range(1, n_episodes+1):
        #RESET
        state = env.reset()
        agent.reset()

        score = 0
        timestep = time.time()
        for t in range(max_t):
            actions = agent.act(state)

            next_state, reward, done = env.step(actions[0])
            losses = agent.step(state, actions, reward, next_state, done, t)
            score += reward
            state = next_state
            if done:
                break

        solved_deque.append(int(score > 0))
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        solv_avg = np.mean(solved_deque)
        scores.append(score_average)

        logger.scalar_summary({'avg_reward': score_average, 'loss_actor': losses[0], 'loss_critic':losses[1]},i_episode)
        if i_episode % save_every == 0: agent.hard_update()
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}, Solv: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep, solv_avg),end="\n")

        if solv_avg >= best and len(solved_deque) >= 100:
            print('SAVED')
            best = solv_avg
            save_model()

    return scores

def test(directory, n_episodes=100, max_t=10, print_every=1):

    actor_path = os.path.join(directory,'checkpoint_actor.pth')
    critic_path = os.path.join(directory,'checkpoint_critic.pth')
    agent.load(actor_path,critic_path)
    scores_deque = deque(maxlen=20)
    scores = []

    solved = 0.0
    avg_steps = 0.0

    for i_episode in range(1, n_episodes+1):
        #RESET
        state = env.reset()
        score = 0
        timestep = time.time()
        for t in range(max_t):
            actions = agent.act(state,add_noise=False)
            next_state, reward, done = env.step(actions[0])
            score += reward
            state = next_state
            if reward > 0.25:
                avg_steps += t+1
                solved += 1
                break

        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_steps/solved)), end="\n")
    return scores

################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=False,help='model to load')
parser.add_argument('--test', action='store_const',const=True,default=False,help='testing flag')
opt = parser.parse_args()
################################################################################################

if __name__ == '__main__':

    if opt.test:
        scores = test(opt.load)
    else:
        scores = train()


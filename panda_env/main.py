#!/usr/bin/env python

import time
import os
import numpy as np
from collections import deque
import argparse

import torch

#CUSTOM MODULES
from env import World
from model import Model

################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=False,help='model to load')
parser.add_argument('--test', action='store_const',const=True,default=False,help='testing flag')
opt = parser.parse_args()
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
from logger import Logger
env = World()
agent = Model()

##########################################################################
#TRAIN THE VISOR
def train(n_episodes=10000, max_t=10, print_every=1, save_every=10):

    logger = Logger('./logs')
    scores_deque = deque(maxlen=20)
    scores= []
    best = 0

    for i_episode in range(1,n_episodes+1):
        state = env.reset()
        score = 0
        timestep = time.time()

        for t in range(max_t):
            #take one step in the environment using the action
            actions = agent.select_action(state)
            next_state,reward,done = env.step_1_6(actions)

            #get the reward for applying action on the prv state
            score += reward

            #store transition into memory (s,a,s_t+1,r)
            agent.memory.push(state.copy(),actions,next_state.copy(),reward,done)
            state = next_state

            #optimize the network using memory
            loss = agent.optimize()

            #stopping condition
            if done:
                break

        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores.append(score_average)

        #LOG THE SUMMARIES
        logger.scalar_summary({'avg_reward': score_average, 'loss': loss},i_episode)

        #update the value network
        if i_episode % save_every == 0:
            agent.target_net.load_state_dict(agent.model.state_dict())

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")

        if score_average >= best and len(scores_deque) == 20:
            print('SAVED')
            best = score_average
            agent.save()

def test(directory, n_episodes=100, max_t=10, print_every=1):

    actor_path = os.path.join(directory,'DQN_1_6.pth')
    agent.load(actor_path)
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
            #take one step in the environment using the action
            actions = agent.select_action(state)
            next_state,reward,done = env.step_1_6(actions)

            #get the reward for applying action on the prv state
            score += reward

            #store transition into memory (s,a,s_t+1,r)
            state = next_state

            #stopping condition
            if done:
                avg_steps += t+1
                solved += 1
                break

        if solved > 0: avg_step = avg_steps / solved
        else: avg_step = 0
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")

    return scores

##########################################################################

if __name__ == '__main__':
    if opt.test:
        scores = test(opt.load)
    else:
        scores = train()


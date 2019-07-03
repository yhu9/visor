#!/usr/bin/env python

import time
import os
import numpy as np
from collections import deque
import argparse
from itertools import count

import torch
import matplotlib.pyplot as plt

#CUSTOM MODULES
from env import World
from model import Model

################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=False,help='model to load')
parser.add_argument('--out',type=str, default='DQN.pth',help='output file')
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
    solved_deque = deque(maxlen=100)
    scores= []
    best = 0

    for i_episode in count():
        state = env.reset()
        score = 0
        timestep = time.time()

        for t in range(max_t):
            #take one step in the environment using the action
            actions = agent.select_action(state)
            next_state,reward,done,_ = env.step_1_6(actions)

            #get the reward for applying action on the prv state
            score += reward

            #store transition into memory (s,a,s_t+1,r)
            agent.memory.push(state,actions,next_state,reward,done)
            state = next_state

            #optimize the network using memory
            loss = agent.optimize()

            #stopping condition
            if done:
                break

        solved_deque.append(int(score > 0))
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        solv_avg = np.mean(solved_deque)
        scores.append(score_average)

        #LOG THE SUMMARIES
        logger.scalar_summary({'avg_reward': score_average, 'loss': loss},i_episode)

        #update the value network
        if i_episode % save_every == 0:
            agent.target_net.load_state_dict(agent.model.state_dict())

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}, Solv: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep, solv_avg),end="\n")

        if solv_avg >= best and len(solved_deque) >= 100:
            print('SAVED')
            best = solv_avg
            agent.save(opt.out)

def test(model_file, n_episodes=200, max_t=20, print_every=1):

    agent.load(model_file)
    scores_deque = deque(maxlen=20)
    scores = []
    best_scores = []

    solved = 0.0
    avg_steps = 0.0

    for i_episode in range(1, n_episodes):
        #RESET
        state = env.reset(manual_pose=i_episode)
        score = 0
        best = -1
        timestep = time.time()
        flag = True
        for t in range(max_t):
            #take one step in the environment using the action
            actions = agent.select_greedy(state)
            next_state,reward,done,threshold = env.step_1_6(actions)

            #get the reward for applying action on the prv state
            score += reward
            if threshold > best:
                best = threshold

            #store transition into memory (s,a,s_t+1,r)
            state = next_state

            #stopping condition
            if done and flag:
                flag = False
                avg_steps += t+1
                solved += 1

        if solved > 0: avg_step = avg_steps / solved
        else: avg_step = 0

        best_scores.append(best)
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")

    return scores,best_scores

##########################################################################

if __name__ == '__main__':
    if opt.test:
        scores, best_scores = test(opt.load)
        scores.sort()
        best_scores.sort()
        np.save(os.path.splitext(opt.load)[0]+ '_score.npy',best_scores)
    else:
        scores = train()


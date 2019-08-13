#!/usr/bin/env python
import time
import os
import numpy as np
from collections import deque
import argparse
from itertools import count
import gc

import torch
import matplotlib.pyplot as plt

#CUSTOM MODULES
from env import World

################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_const',const=True,default=True,help='testing flag')
parser.add_argument('--noise', type=float,default=0.00,help='noise value. default is 0.00')
opt = parser.parse_args()
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
env = World()

##########################################################################
def test(n_episodes=200, max_t=20, print_every=1,noise1=0.00,noise2=0.00):
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
        #flag = True
        for t in range(max_t):
            #take one step in the environment using the action
            reward,done = env.calcVisor(light_noise=noise2,geom_noise=noise1)

            #get the reward for applying action on the prv state
            score += reward
            if reward > 0.25:
                avg_steps += t+1
                solved += 1

            if reward > best:
                best = reward

            break

        if solved > 0: avg_step = avg_steps / solved
        else: avg_step = 0

        best_scores.append(best)
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        #if i_episode % print_every == 0:
        #    print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
        #          .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")
    print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
          .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")
    #return scores,best_scores
    return solved / i_episode

##########################################################################

if __name__ == '__main__':
    if opt.test:
        lnoise_vals = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
        gnoise_vals = [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045]
        for i,j in zip(lnoise_vals,gnoise_vals):
            vals = []
            for _ in range(5):
                score = test(noise1=j,noise2=i)
                vals.append(score)
            print('mode: ' + str(i) + ' mean score: ' + str(np.mean(vals)))
    else:
        scores = train()






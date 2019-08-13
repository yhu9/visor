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
opt = parser.parse_args()
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
env = World()

##########################################################################
def test(n_episodes=200, max_t=20, print_every=1):
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
            #visor, s2 ,reward,done = env.step2_4(actions)
            reward,done = env.calcVisor()

            #get the reward for applying action on the prv state
            score += reward
            #stopping condition
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
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")
    return scores,best_scores

##########################################################################

if __name__ == '__main__':
    if opt.test:
        scores, best_scores = test()
        scores.sort()
        best_scores.sort()
        np.save('baseline_score.npy',best_scores)
    else:
        scores = train()


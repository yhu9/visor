#!/usr/bin/env python
import time
import os
import numpy as np
from collections import deque
import argparse
from itertools import count

#CUSTOM MODULES
from env import World
from model2 import Model

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
agent = Model(opt.load,mode='DDQN')
##########################################################################
#TRAIN THE VISOR
def train(n_episodes=40000, max_t=10,  save_every=20):

    logger = Logger('./logs')
    scores_deque = deque(maxlen=200)
    solved_deque = deque(maxlen=200)
    scores= []
    best = 0

    for i_episode in count():
        state = env.reset2_4(manual_pose=(i_episode % 200) + 1)
        score = 0
        timestep = time.time()
        for t in range(max_t):
            #take one step in the environment using the action
            actions = agent.select_action(state)
            visor,s2,r1,r2,done = env.step2_4(actions)
            next_state = (visor,s2)
            score += r2

            #store HER transition into memory (s,a,s_t+1,r)
            sg1 = state[1].copy()
            sg1[:,:,-1] = s2[:,:,-1]
            agent.memory.push((visor,sg1),actions,next_state,r1,done)

            #push state and goal to memory
            agent.memory.push(state,actions,next_state,r2,done)
            state = next_state

            #optimize the network using memory
            loss = agent.optimize()

            #stopping condition
            if done:
                break

        solved_deque.append(done and r2 > 0)
        scores_deque.append(r2)
        score_average = np.mean(scores_deque)
        solv_avg = np.mean(solved_deque)
        scores.append(score_average)

        #LOG THE SUMMARIES
        logger.scalar_summary({'avg_reward': score_average, 'loss': loss},i_episode)

        #update the value network
        if i_episode % save_every == 0:
            agent.target_net.load_state_dict(agent.model.state_dict())

        print('\rEpisode {}, Average Score: {:.2f}, Time: {:.2f}, Solv: {:.2f}, Loss: {:.4f}, Best: {:.2f}'\
              .format(i_episode, score_average, time.time() - timestep, solv_avg, loss, best),end="\n")

        if solv_avg >= best and len(solved_deque) >= 200:
            print('SAVED')
            best = solv_avg
            agent.save(opt.out)
    agent.save(opt.out)

#TEST THE AGENT
def test(n_episodes=200, max_t=20):

    scores_deque = deque(maxlen=20)
    scores = []
    best_scores = []

    solved = 0.0
    avg_steps = 0.0

    for i_episode in range(1, n_episodes):
        state = env.reset2_4(manual_pose=i_episode)
        score = 0
        best = -1
        timestep = time.time()

        for t in range(max_t):
            #take one step in the environment using the action
            actions = agent.select_greedy(state)
            visor, s2 ,r1,r2 ,done = env.step2_4(actions)
            next_state = (visor,s2)
            state = next_state

            #get the reward for applying action on the prv state
            score += r2

            #stopping condition
            if r2> 0.25:
                avg_steps += t+1
                solved += 1
            if r2 > best:
                best =r2
            if done: break

        if solved > 0: avg_step = avg_steps / solved
        else: avg_step = 0

        best_scores.append(best)
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)

        print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Solved: {:.2f}, AvgStps: {:.2f}'\
              .format(i_episode, score_average, np.max(scores), np.min(scores), (solved/i_episode),(avg_step)), end="\n")
    return scores,best_scores

##########################################################################
if __name__ == '__main__':
    if opt.test:
        scores, best_scores = test()
        scores.sort()
        best_scores.sort()
        np.save(os.path.splitext(opt.load)[0]+ '_score.npy',best_scores)
    else:
        scores = train()


#!/usr/bin/env python

import time
import random
from collections import deque
import matplotlib.pyplot as plt
from itertools import count
import sys
import os
from math import pi, sin, cos,sqrt
import numpy as np
import argparse

#CUSTOM MODULES
from model import Model
from utils import ShadowDetector
from utils import LM_Detector

#OTHER LIBRARIES
import cv2
import torch

#PANDA3D
from panda3d.core import *
from direct.task import Task
import direct.directbase.DirectStart
#from direct.interval.IntervalGlobal import *
from direct.gui.DirectGui import OnscreenText
from direct.showbase.DirectObject import DirectObject
from direct.actor.Actor import Actor
from direct.filter.FilterManager import FilterManager

from panda3d.direct import throw_new_frame
################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=False,help='model to load')
parser.add_argument('--test', action='store_const',const=True,default=False,help='testing flag')

opt = parser.parse_args()
################################################################################################

#OBSERVER IN OUR ENVIRONMENT WHICH DEFINES THE REWARD FUNCTION
class Observer():
    def __init__(self):
        #EVALUATOR
        self.shadowfunc = ShadowDetector()
        self.lmfunc = LM_Detector()

    #GET THE CURRENT FRAME AS NUMPY ARRAY
    def getFrame_notex(self):
        base.graphicsEngine.renderFrame()
        dr = base.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        v = memoryview(data).tolist()
        img = np.array(v,dtype=np.uint8)
        img = img.reshape((tex.getYSize(),tex.getXSize(),4))
        img = img[:,:,:3]
        img = img[::-1]
        img = img[159:601,150:503,:]        #boundry based on range of head motion
        img = cv2.resize(img,(112,112))
        return img / 255.0

    #GET THE CURRENT FRAME AS NUMPY ARRAY
    def getFrame(self):
        base.graphicsEngine.renderFrame()
        dr = base.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        v = memoryview(data).tolist()
        img = np.array(v,dtype=np.uint8)
        img = img.reshape((tex.getYSize(),tex.getXSize(),4))
        img = img[:,:,:3]
        img = img[::-1]
        img, lm = self.lmfunc.get_lm(img)
        h,w = img.shape[:2]
        img = cv2.resize(img,(112,112))
        lm[:,0] = lm[:,0] * (112 / w)
        lm[:,1] = lm[:,1] * (112 / h)

        return lm, img / 255.0


    #REWARD MAP GENERATION GIVEN STATE S
    def genRewardGT(self,params,noshadow_img,shadow_img):

        noshadow = (noshadow_img * 255).astype(np.uint8)
        shadow = (shadow_img * 255).astype(np.uint8)

        eye_mask = self.lmfunc.get_eyesGT(noshadow)     #((cx,cy),(h,w),rotation)
        shadow_mask = self.shadowfunc.get_shadowgt(shadow)
        lm = np.zeros((5,2)).astype(np.uint8)

        IOU = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(np.logical_or(eye_mask,shadow_mask))
        EYE = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(eye_mask)
        #shadow_mask = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(shadow_mask)
        #525 = 15 * 35 which is the area of the visor roughly
        #params are the visor parameters (x,y,w,h,theta)
        #A = (params[2] * params[3]) / np.sum(eye_mask)
        #threshold = EYE + IOU
        reward = IOU

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(params,eye_mask,shadow_mask,shadow.copy(),lm,IOU,EYE,reward)

        return reward

        #reward
        if threshold > 1.2:
            reward,flag = threshold, True
        else:
            reward,flag = -0.1, False

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(params,eye_mask,shadow_mask,shadow.copy(),lm,IOU,EYE,A,reward)

        return reward,flag

    #REWARD MAP GENERATION GIVEN STATE S
    def genReward(self,params,rgb,lm,gt=False):

        if gt:
            eye_mask = self.lmfunc.get_eyesGT(rgb)     #((cx,cy),(h,w),rotation)
            shadow = self.shadowfunc.get_shadowgt(rgb)
            lm = np.zeros((5,2)).astype(np.uint8)
        else:
            eye_mask = self.lmfunc.get_eyes(rgb,lm)     #((cx,cy),(h,w),rotation)
            shadow = self.shadowfunc.get_shadow(rgb)

        IOU = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(np.logical_or(eye_mask,shadow))
        EYE = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(eye_mask)
        #SHADOW = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(shadow)

        #525 = 15 * 35 which is the area of the visor roughly
        #params are the visor parameters (x,y,w,h,theta)
        A = (params[2] * params[3]) / np.sum(eye_mask)
        threshold = EYE + IOU

        #reward
        if threshold > 1.3:
            reward,flag = torch.Tensor([100.0]), True
        else:
            reward,flag = torch.Tensor([-1.00]), False

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(params,eye_mask,shadow,rgb.copy(),lm,IOU,EYE,reward)

        return reward,flag

    def drawReward(self,params,eye_mask,shadow,rgb,lm,IOU,EYE,reward):

        h,w,d = rgb.shape
        img = np.zeros((h,w+100,d))

        rgb[eye_mask] = rgb[eye_mask] * [0,0,1]
        rgb[shadow] = rgb[shadow] * [1,0,0]

        IOU = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(np.logical_or(eye_mask,shadow))
        EYE = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(eye_mask)

        #show landmarks
        for x,y in lm:
            cv2.circle(rgb,(x,y),2,(255,0,255),-1)

        #IMG IS 224 X 112
        cv2.putText(img,"IOU %.3f" % IOU,(2,h-54),2,0.4,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,"EYE %.3f" % EYE,(2,h-34),2,0.4,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,"reward %.3f" % reward ,(2,h-14),2,0.4,(0,255,0),1,cv2.LINE_AA)

        img[:,100:,:] = rgb
        cv2.imshow('semantic mask',img)
        cv2.waitKey(10)

################################################################################################
################################################################################################
################################################################################################
################################################################################################

# Function to put instructions on the screen.
def addInstructions(pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=base.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)

# Function to put title on the screen.
def addTitle(text):
    return OnscreenText(text=text, style=1, fg=(1, 1, 1, 1), scale=.07,
                        parent=base.a2dBottomRight, align=TextNode.ARight,
                        pos=(-0.1, 0.09), shadow=(0, 0, 0, 1))

#MAIN CLASS FOR GENERATING THE ENVIRONMENT
class World(DirectObject):
    def __init__(self):

        self.obs = Observer()
        # Preliminary capabilities check.
        if not base.win.getGsg().getSupportsBasicShaders():
            self.t = addTitle(
                "Shadow Demo: Video driver reports that shaders are not supported.")
            return
        if not base.win.getGsg().getSupportsDepthTexture():
            self.t = addTitle(
                "Shadow Demo: Video driver reports that depth textures are not supported.")
            return

        #base.backfaceCullingOff()
        base.setBackgroundColor(0, 0, 0, 1)
        base.camLens.setNearFar(0.1, 10000)
        base.camLens.setFov(60)

        # examine the state space
        self.state_size = (6,224,112)
        print('Size of state:', self.state_size)
        self.action_low = np.array([0,0,0,0,0])
        print('Action low:', self.action_low)
        self.action_high = np.array([35,15,35,15,3.14/2])
        print('Action high: ', self.action_high)

        #initialize the scene
        self.init_scene()
        self.incrementCameraPosition(0)
        self.incLightPos(0)

    #RESET THE ENVIRONMENT
    def reset(self):
        self.dennis.pose('head_movement',random.randint(1,200))     #CHOOSE A RANDOM POSE
        self.light_angle = random.randint(-5,5)
        self.incLightPos(0)                                         #PUT LIGHT IN RANDOM POSITION

        #get image without shadow
        self.shadowoff()
        self.noshadow_img = self.obs.getFrame_notex()
        self.prv_frame = self.noshadow_img.copy()

        #init the visor to the same position always
        self.visorparam = [17,7,15,10,0]                              #x,y,w,h,r              #INITIAL VISOR POSITION IS ALWAYS THE SAME
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                if self.visormask[j,i] == 1:
                    self.hexes[i][j].show(BitMask32.bit(1))
                else:
                    self.hexes[i][j].hide(BitMask32.bit(1))
        render.show(BitMask32.bit(1))

        #render the reset frame
        base.graphicsEngine.renderFrame()
        cur_frame = env.obs.getFrame_notex()
        h,w = cur_frame.shape[:2]
        frame = np.zeros((h,w,6))
        frame[:,:,:3] = self.prv_frame
        frame[:,:,3:] = cur_frame

        state = frame,self.visorparam
        return state

    #INITIALIZE THE 3D ENVIRONMENT
    def init_scene(self):
        # Load the scene.
        floorTex = loader.loadTexture('maps/envir-ground.jpg')

        cm = CardMaker('')
        cm.setFrame(-2, 2, -2, 2)
        floor = render.attachNewNode(PandaNode("floor"))
        for y in range(12):
            for x in range(12):
                nn = floor.attachNewNode(cm.generate())
                nn.setP(-90)
                nn.setPos((x - 6) * 4, (y - 6) * 4, 0)
        floor.setTexture(floorTex)
        floor.flattenStrong()

        self.car = loader.loadModel("assets/car.egg")
        self.car.reparentTo(render)
        self.car.setPos(0, 0, 0)
        #self.car.set_two_sided(True)    #BEST IF I CAN SOLVE THE BIAS PROBLEM ON THE SHADER

        self.dennis = Actor('assets/dennis.egg',{"head_movement": "assets/dennis-head_movement.egg"})
        self.dennis.reparentTo(self.car)
        self.dennis.pose('head_movement',1)
        #self.dennis.setPlayRate(1.,'head_movement')
        #self.dennis.loop("head_movement")
        #CURRENTLY SHADOW QUALITY IS REDUCED DUE TO SHADOW ACNE

        self.visor, self.hexes = self.genVisor()
        self.visor.reparentTo(self.car)
        self.visor.set_two_sided(True)
        self.visor.setPos(-3.75,.5,2.6)
        self.visor.setH(-90)
        self.visor.setP(70)
        self.visor.setScale(0.015,0.015,0.015)

        #LOAD THE LIGHT SOURCE
        self.sun = DirectionalLight("Dlight")
        self.sun.color = self.sun.color * 5
        self.light = render.attachNewNode(self.sun)
        self.light.node().setScene(render)
        self.light.node().setShadowCaster(True)
        self.light.node().setCameraMask(BitMask32.bit(1))
        self.light.node().showFrustum()
        self.light.node().getLens().set_film_size(20)
        self.light.node().getLens().setFov(20)
        self.light.node().getLens().setNearFar(10, 50)
        render.setLight(self.light)

        self.alight = render.attachNewNode(AmbientLight("Ambient"))
        self.alight.node().setColor(LVector4(0.2, 0.2, 0.2, 1))
        render.setLight(self.alight)

        #Important! Enable the shader generator.
        #render.setShaderInput('push',0.10)
        render.setShaderAuto()
        render.show(BitMask32.bit(1))

        # default values
        self.max_reward = -10.0
        self.episode = 0
        self.light_angle = 0.0
        self.car_x = 0.0
        self.cameraSelection = 0
        self.visorMode = 0
        self.visormask = np.zeros((15,35))
        self.visorparam = [17,7,10,8,0]      #x,y,w,h,r
        self.light.node().hideFrustum()

    def shadowoff(self):
        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                self.hexes[i][j].hide(BitMask32.bit(1))
        render.show(BitMask32.bit(1))

    def shadowon(self):
        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                if self.visormask[j,i] == 1:
                    self.hexes[i][j].show(BitMask32.bit(1))
                else:
                    self.hexes[i][j].hide(BitMask32.bit(1))
        render.show(BitMask32.bit(1))

    #get state
    def getstate(self,prv_frame,cur_frame):
        h,w = cur_frame.shape[:2]
        d = 6
        frame = np.zeros((h,w,d))
        frame[:,:,:3] = prv_frame
        frame[:,:,3:] = cur_frame
        state = frame,self.visorparam
        return state

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def step(self,actions,speed=1):

        actions = (actions + 1) * (self.action_low + self.action_high) / 2
        self.visorparam = actions

        #get image with shadow after action
        self.incLightPos(speed)
        self.visorparam[0] = min(max(0,self.visorparam[0]),34)
        self.visorparam[1] = min(max(0,self.visorparam[1]),14)
        self.visorparam[2] = min(max(0,self.visorparam[2]),34)
        self.visorparam[3] = min(max(0,self.visorparam[3]),14)
        self.visorparam[4] = min(max(-pi,self.visorparam[4]),pi)
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #display the visor and show some data
        self.shadowon()
        cur_frame = self.obs.getFrame_notex()
        cv2.imshow('prv',(self.prv_frame * 255).astype(np.uint8))
        cv2.imshow('cur',(cur_frame*255).astype(np.uint8))
        cv2.imshow('visormask',cv2.resize((self.visormask * 255).astype(np.uint8), (35*10,15*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(1)

        #get next state and reward
        reward = self.obs.genRewardGT(self.visorparam,self.noshadow_img,cur_frame)

        #set the next state
        next_state = self.getstate(self.prv_frame,cur_frame)
        self.prv_frame = cur_frame.copy()

        return next_state,reward,False

    def spinLightTask(self,task):
        angleDegrees = (task.time * 5 - 90)
        angleRadians = angleDegrees * (pi / 180.0)
        self.light.setPos(20.0 * sin(angleRadians),20.0 * cos(angleRadians),10)
        self.light.lookAt(0,0,0)

        return task.cont

    def incLightPos(self,speed):
        angleDegrees = (self.light_angle - 80)
        angleRadians = angleDegrees * (pi / 180.0)
        self.light.setPos(20.0 * sin(angleRadians),20.0 * cos(angleRadians),10)
        self.light.lookAt(0,0,0)
        self.light_angle = self.light_angle + speed

    def incCarPos(self,speed):
        self.car_x += (self.car_x + speed) % 180
        self.car.setY(sin((self.car_x)* pi / 180) * 0.1 )

    def incrementCameraPosition(self,n):
        self.cameraSelection = (self.cameraSelection + n) % 3
        if (self.cameraSelection == 1):
            base.cam.setPos(-20,0,5)
            base.cam.lookAt(0,0,0)

        if (self.cameraSelection == 0):
            #base.camLens.setNearFar(0,10)
            base.cam.setPos(-3.8,0.0,2.25)
            base.cam.lookAt(-3,-0.2,2.69)

        if (self.cameraSelection == 2):
            base.cam.setPos(self.light.getPos())
            base.cam.lookAt(0,0,0)

    def genVisor(self,w=35,h=15):
        visor = render.attach_new_node("visor")
        objects = [[None] * h for i in range(w)]
        offsetx = (1.55) / 2.00
        offsety = (sqrt(1 - (offsetx * offsetx)) + 1) / 2.00
        x,y = 0,0
        for i in range(0,w):
            for j in range(0,h):
                objects[i][j] = loader.loadModel("assets/hex/hexagon.egg")
                objects[i][j].reparentTo(visor)
                objects[i][j].setPos((offsety * 2.0*i) + (offsety * (j % 2)), offsetx*2*j,5)
                objects[i][j].setScale(1.1,1.1,1.1)
                objects[i][j].setAlphaScale(0.01)

        return visor,objects

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

logger = Logger('./logs')
agent = Agent(action_size=action_size, random_seed=random_seed)
def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'model/checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'model/checkpoint_critic.pth')

def ddpg(n_episodes=1000, max_t=10, print_every=1, save_every=10):
    scores_deque = deque(maxlen=20)
    scores = []
    best = 0

    for i_episode in range(1, n_episodes+1):
        #RESET
        state = env.reset()
        agent.reset()

        score = 0
        timestep = time.time()
        for t in range(max_t):
            actions = agent.act(state)

            next_state, reward, done = env.step(actions[0])
            done = t == max_t - 1

            losses = agent.step(state, actions, reward, next_state, done, t)
            score += reward
            state = next_state
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)

        logger.scalar_summary({'avg_reward': score_average, 'loss_actor': losses[0], 'loss_critic':losses[1]},i_episode)

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")

        if score_average >= best:
            best = score_average
            save_model()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))

    return scores

scores = ddpg()




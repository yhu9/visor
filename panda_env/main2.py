#!/usr/bin/env python

from itertools import count
import sys
import os
from math import pi, sin, cos,sqrt
from random import *
import numpy as np
import argparse

#CUSTOM MODULES
#from model import Model
from model2 import Model

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

        #initialize the scene
        self.init_scene()
        self.incrementCameraPosition(0)
        self.incLightPos(0)

        #ADD TASKS
        #1. SPINS THE LIGHT SOURCE
        #2. VISOR CONTROLLER
        taskMgr.add(self.spinLightTask,"SpinLightTask")        #ROTATE THE DIRECTIONAL LIGHTING SOURCE
        if not opt.test:
            taskMgr.doMethodLater(0.1,self.trainVisor2,'training control')
        elif opt.load:
            self.time_taken = []
            self.reward = 0.0
            self.success = 0
            self.failure = 0
            taskMgr.doMethodLater(0.1,self.testVisor,'testing control')

        #define our policy network
        self.net = Model(load=opt.load)

    #RESET THE ENVIRONMENT
    def reset(self):
        self.dennis.pose('head_movement',randint(1,200))     #CHOOSE A RANDOM POSE
        self.light_angle = randint(-5,5)
        self.incLightPos(0)                                         #PUT LIGHT IN RANDOM POSITION

        #init the visor to the same position always
        self.visorparam = [17,7,5,4,0]                              #x,y,w,h,r              #INITIAL VISOR POSITION IS ALWAYS THE SAME
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
        self.alight.node().setColor(LVector4(0.5, 0.5, 0.5, 1))
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
        img, lm = self.net.lmfunc.get_lm(img)
        h,w = img.shape[:2]
        img = cv2.resize(img,(112,224))
        lm[:,0] = lm[:,0] * (112 / w)
        lm[:,1] = lm[:,1] * (224 / h)

        return lm, img / 255.0

    #get state
    def getstate(self,prv_frame,cur_frame):
        h,w = cur_frame.shape[:2]
        d = 6
        state = np.zeros((h,w,d))
        state[:,:,:3] = prv_frame
        state[:,:,3:] = cur_frame

        state = torch.from_numpy(np.ascontiguousarray(state)).float()
        state = state.permute(2,0,1)
        state = state.unsqueeze(0)
        return state

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def step(self,action):

        #save previous frame
        prv_lm, prv_frame = self.getFrame()

        #GET THE NEW VISOR MASK
        for i,a in enumerate(action):
            self.visorparam[i] += a

        self.visorparam[0] = min(max(0,self.visorparam[0]),34)
        self.visorparam[1] = min(max(0,self.visorparam[1]),14)
        self.visorparam[2] = min(max(0,self.visorparam[2]),34)
        self.visorparam[3] = min(max(0,self.visorparam[3]),14)
        self.visorparam[4] = min(max(-pi,self.visorparam[4]),pi)
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])     #PADDED HEIGHT AND WIDTH OF 15PIXELS
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

        #display the visor
        cv2.imshow('visormask',cv2.resize((self.visormask * 255).astype(np.uint8), (35*10,15*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(10)

        #render the frame
        self.incLightPos(1)
        base.graphicsEngine.renderFrame()

        #save new frame
        cur_lm, cur_frame = self.getFrame()

        #get the reward for applying action on the prv state
        reward,done = self.net.genReward(self.visorparam,cur_frame,cur_lm)
        if not done:
            next_state = self.getstate(prv_frame,cur_frame)
        else:
            next_state = None

        return next_state,reward,done

    def spinLightTask(self,task):
        angleDegrees = (task.time * 5 - 90)
        angleRadians = angleDegrees * (pi / 180.0)
        self.light.setPos(20.0 * sin(angleRadians),20.0 * cos(angleRadians),10)
        self.light.lookAt(0,0,0)

        return task.cont

    def incLightPos(self,speed):
        angleDegrees = (self.light_angle - 90)
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

    ##########################################################################
    #TRAIN THE VISOR
    def trainVisor2(self,task):

        #RESET
        self.reset()

        #GET THE CURRENT FRAME AS A NUMPY ARRAY
        cur_lm, cur_frame = self.getFrame()
        prv_lm, prv_frame = self.getFrame()
        state = self.getstate(prv_frame,cur_frame)

        accum_reward = 0.0
        for t in count():

            #NOISY ACTION SELECTION FOR A SINGLE STATE
            actions = self.net.select_action(state)

            #take one step in the environment using the action
            next_state,reward,done = self.step(actions[0])
            accum_reward += reward

            #store transition into memory if it is not an end state (s,a,s_t+1,r)
            if not done:
                self.net.memory.push(state,actions,next_state,reward)

            #UPDATE PREVIOUS STATE
            state = next_state

            #optimize the network using memory
            loss = self.net.optimize()

            #VISUALIZE TRAINING ON TERMINA
            sys.stdout.write("episode: %i | loss: %.5f |  R_step: %.5f |    R_accum: %.5f | R_max: %.5f \r" %(self.episode,loss,reward.item(),accum_reward,self.max_reward))

            #stopping condition
            if done or t == 10:
                if accum_reward > self.max_reward:
                    self.max_reward = accum_reward
                    self.net.save()
                break

        #LOG THE SUMMARIES
        self.net.logger.scalar_summary({'max_reward': self.max_reward, 'ep_reward': accum_reward, 'loss': loss},self.episode)

        #update the value network
        self.episode += 1
        if self.episode % self.net.TARGET_UPDATE == 0:
            self.net.target_actor.load_state_dict(self.net.actor.state_dict())
            self.net.target_critic.load_state_dict(self.net.critic.state_dict())

        return task.again

    ##########################################################################
    #TEST THE VISOR
    def testVisor(self,task):

        #RESET
        self.reset()

        #GET THE CURRENT FRAME AS A NUMPY ARRAY
        cur_lm, cur_frame = self.getFrame()
        prv_lm, prv_frame = self.getFrame()
        state = self.getstate(prv_frame,cur_frame)

        accum_reward = 0.0
        for t in count():
            #get the immediate reward for current state
            reward,done = self.net.genReward(self.visorparam,cur_frame,cur_lm)
            accum_reward += reward

            #take one step in the environment using the action
            a1,a2,a3 = self.net.select_greedy(state)
            self.takeaction(a1.item(),a2.item(),a3.item())

            #get new state
            if not done:
                prv_lm, prv_frame = cur_lm.copy(),cur_frame.copy()
                cur_lm, cur_frame = self.getFrame()
                next_state = self.getstate(prv_frame,cur_frame)
            else:
                next_state = None

            #store transition into memory (s,a,s_t+1,r)
            #self.net.memory.push(state,a1,a2,a3,next_state,reward)
            state = next_state

            #display the visor and show some data
            cv2.imshow('visormask',cv2.resize((self.visormask * 255).astype(np.uint8), (35*10,15*10), interpolation = cv2.INTER_LINEAR))
            cv2.waitKey(10)

            #stopping condition
            if done:
                self.success += 1
                break
            elif t == 20:
                self.failure += 1
                break

        self.episode += 1
        self.reward += accum_reward
        self.time_taken.append(t)

        success_rate = self.success / self.episode
        failure_rate = self.failure / self.episode
        avg_reward = self.reward / self.episode
        avg_time = sum(self.time_taken) / len(self.time_taken)
        sys.stdout.write("episodes: %i | success_rate: %.5f | failure_rate: %.5f | avg_time: %.5f | avg_reward: %.5f \r" %(self.episode,success_rate,failure_rate,avg_time,avg_reward))

        if self.episode == 100: return task.done
        return task.again
    ##########################################################################

w = World()
base.run()




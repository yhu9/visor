#!/usr/bin/env python

from panda3d.core import *
import sys
import os
from math import pi, sin, cos,sqrt

from direct.task import Task
import direct.directbase.DirectStart
from direct.interval.IntervalGlobal import *
from direct.gui.DirectGui import OnscreenText
from direct.showbase.DirectObject import DirectObject
from direct.actor import Actor
from random import *

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


        base.setBackgroundColor(0, 0, 0.2, 1)
        base.camLens.setNearFar(1.0, 10000)
        base.camLens.setFov(75)
        base.disableMouse()

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

        self.car = loader.loadModel('assets/my_model.egg')
        self.car.reparentTo(render)
        self.car.setPos(0, 0, 0)
        self.car.set_two_sided(False)

        self.visor, self.hexes = self.genVisor()
        self.visor.reparentTo(self.car)
        self.visor.set_two_sided(True)
        self.visor.setPos(-3.75,.5,2.6)
        self.visor.setH(-90)
        self.visor.setP(70)
        self.visor.setScale(0.015,0.015,0.015)

        #LOAD THE LIGHT SOURCE
        self.sun = Spotlight("Spot")
        self.sun.color = self.sun.color * 5
        self.light = render.attachNewNode(self.sun)
        #self.light = render.attachNewNode(DirectionalLight("Spot"))
        self.light.node().setScene(render)
        self.light.node().setShadowCaster(True)
        self.light.node().setCameraMask(BitMask32.bit(1))
        self.light.node().showFrustum()
        self.light.node().getLens().setFov(40)
        self.light.node().getLens().setNearFar(10, 100)
        render.setLight(self.light)

        self.alight = render.attachNewNode(AmbientLight("Ambient"))
        self.alight.node().setColor(LVector4(0.5, 0.5, 0.5, 1))
        render.setLight(self.alight)

        # Important! Enable the shader generator.
        render.setShaderAuto()
        render.show(BitMask32.bit(1))

        #ADD TASKS
        #1. SPINS THE LIGHT SOURCE
        #2. VISOR CONTROLLER
        taskMgr.add(self.spinLightTask,"SpinLightTask")        #ROTATE THE DIRECTIONAL LIGHTING SOURCE
        taskMgr.doMethodLater(0.5,self.randomVisor,'random controller')

        # default values
        self.cameraSelection = 0
        self.lightSelection = 0
        self.visorMode = 0

        #SOME CONTROL SEQUENCES
        self.inst_p = addInstructions(0.06, 'up/down arrow: zoom in/out')
        self.inst_x = addInstructions(0.12, 'Left/Right Arrow : switch camera angles')
        addInstructions(0.18, 'a : put sun on  face')
        addInstructions(0.24, 's : toggleSun')
        addInstructions(0.30, 'd : toggle visor')
        addInstructions(0.36, 'v: View the Depth-Texture results')
        addInstructions(0.42, 'tab : view buffer')
        self.accept('escape', sys.exit)
        self.accept("a", self.putSunOnFace)
        self.accept("s", self.toggleSun,[1])
        self.accept("d", self.toggleVisor,[1])
        self.accept("arrow_up",self.zoom,[-1])
        self.accept("arrow_down",self.zoom,[1])
        self.accept("arrow_left", self.incrementCameraPosition, [-1])
        self.accept("arrow_right", self.incrementCameraPosition, [1])
        self.accept("tab", base.bufferViewer.toggleEnable)
        self.accept("v", base.bufferViewer.toggleEnable)
        self.accept("o", base.oobe)
        self.incrementCameraPosition(0)
        self.z = 5

    def spinLightTask(self,task):
        angleDegrees = (task.time * 5 - 90)
        angleRadians = angleDegrees * (pi / 180.0)
        self.light.setPos(20.0 * sin(angleRadians),20.0 * cos(angleRadians),10)
        self.light.lookAt(0,0,0)

        return task.cont

    def toggleVisor(self,n):
        self.visorMode = (self.visorMode + n) % 2
        if (self.visorMode == 1):
            taskMgr.remove("random controller")
            for row in self.hexes:
                for h in row:
                    h.show(BitMask32.bit(1))
            render.show(BitMask32.bit(1))
        if (self.visorMode == 0):
            taskMgr.doMethodLater(0.5,self.randomVisor,'random controller')
            #taskMgr.add(self.randomVisor,"random controller")

    def toggleSun(self,n):
        self.lightSelection = (self.lightSelection + n) % 2
        if (self.lightSelection == 1):
            taskMgr.remove("SpinLightTask")
        if (self.lightSelection == 0):
            taskMgr.add(self.spinLightTask,"SpinLightTask")

    def putSunOnFace(self):
        self.lightSelection = 0
        self.toggleSun(1)
        self.light.setPos(-20,0,10)
        self.light.lookAt(0,0,0)

    def zoom(self,n):
        x,y,z = base.cam.getPos()
        mag = sqrt(x*x + y*y + z*z)
        base.cam.reparentTo(render)
        base.cam.setPos(x + n*x/mag,y + n*y/mag,z+n*z/mag)
        base.cam.lookAt(0,0,0)

    def incrementCameraPosition(self,n):
        self.cameraSelection = (self.cameraSelection + n) % 3
        if (self.cameraSelection == 0):
            base.cam.reparentTo(render)
            base.cam.setPos(-20,0,5)
            base.cam.lookAt(0,0,0)
            self.light.node().showFrustum()

        if (self.cameraSelection == 1):
            base.cam.reparentTo(render)
            base.cam.setPos(-4.4,0,2.25)
            base.cam.lookAt(-3,-0.2,2.89)
            self.light.node().showFrustum()

        if (self.cameraSelection == 2):
            base.cam.reparentTo(render)
            base.cam.setPos(self.light.getPos())
            base.cam.lookAt(0,0,0)
            self.light.node().showFrustum()

    def randomVisor(self,task):

        for _ in range(300):
            i = randint(0,len(self.hexes)-1)
            j = randint(0,len(self.hexes[-1]) - 1)
            self.hexes[i][j].hide(BitMask32.bit(1))

        for _ in range(300):
            i = randint(0,len(self.hexes)-1)
            j = randint(0,len(self.hexes[-1]) - 1)
            self.hexes[i][j].show(BitMask32.bit(1))

        render.show(BitMask32.bit(1))

        return Task.again

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

w = World()
base.run()

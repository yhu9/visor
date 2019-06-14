
#!/usr/bin/env python

from itertools import count
import sys
from math import pi, sin, cos,sqrt
from random import *
import numpy as np

#CUSTOM MODULES
#from model import Model

#OTHER LIBRARIES
import cv2

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

        #base.backfaceCullingOff()
        #props = WindowProperties()
        #props.setSize(224,224)
        #base.win.requestProperties(props)
        base.setBackgroundColor(0, 0, 0, 1)
        base.camLens.setNearFar(0.1, 10000)
        base.camLens.setFov(60)

        #initialize the scene
        self.init_scene()
        self.incrementCameraPosition(0)
        self.putSunOnFace()

        #ADD TASKS
        #1. SPINS THE LIGHT SOURCE
        #2. VISOR CONTROLLER
        #taskMgr.add(self.spinLightTask,"SpinLightTask")        #ROTATE THE DIRECTIONAL LIGHTING SOURCE
        #taskMgr.doMethodLater(0.5,self.randomVisor,'random controller')
        #if not opt.test:
        #    taskMgr.doMethodLater(0.2,self.trainVisor,'training control')
        #else:
        #    taskMgr.doMethodLater(0.2,self.testVisor,'testing control')

        #add user controller
        self.addControls()

        #define our policy network
        #self.net = Model(load=opt.load)

    #RESET THE ENVIRONMENT
    def reset(self):
        self.cameraSelection = 0
        self.incrementCameraPosition(0)
        self.light_angle = 0.0
        self.dennis.pose('head_movement',80)
        self.putSunOnFace()
        self.visorparam = [17,7,5,4,0]      #x,y,w,h,r
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

        base.graphicsEngine.renderFrame()

    #SOME CONTROL SEQUENCES
    def addControls(self):
        #self.inst_p = addInstructions(0.06, 'up/down arrow: zoom in/out')
        #self.inst_x = addInstructions(0.12, 'Left/Right Arrow : switch camera angles')
        #addInstructions(0.18, 'a : put sun on face')
        #addInstructions(0.24, 's : toggleSun')
        #addInstructions(0.30, 'd : toggle visor')
        #addInstructions(0.36, 'v: View the Depth-Texture results')
        #addInstructions(0.42, 'tab : view buffer')
        self.accept('escape', sys.exit)
        self.accept("a", self.putSunOnFace)
        self.accept("s", self.toggleSun,[1])
        self.accept("d", self.toggleVisor,[1])
        self.accept("r", self.recordScreen)
        self.accept("arrow_up",self.zoom,[-1])
        self.accept("arrow_down",self.zoom,[1])
        self.accept("arrow_left", self.incrementCameraPosition, [-1])
        self.accept("arrow_right", self.incrementCameraPosition, [1])
        self.accept("f12",base.screenshot,['recording/snapshot'])
        self.accept("tab", base.bufferViewer.toggleEnable)
        self.accept("v", base.bufferViewer.toggleEnable)
        self.accept("o", base.oobe)

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
        self.car.set_two_sided(True)    #BEST IF I CAN SOLVE THE BIAS PROBLEM ON THE SHADER

        self.dennis = Actor('assets/dennis.egg',{"head_movement": "assets/dennis-head_movement.egg"})
        self.dennis.reparentTo(self.car)
        self.dennis.setPlayRate(0.5,'head_movement')
        self.dennis.loop("head_movement")
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
        self.light_angle = 0.0
        self.car_x = 0.0
        self.cameraSelection = 0
        self.lightSelection = 0
        self.visorMode = 0
        self.visormask = np.zeros((15,35))
        self.visorparam = [17,7,10,8,0]      #x,y,w,h,r
        self.light.node().hideFrustum()

    #GET THE CURRENT FRAME AS NUMPY ARRAY
    def getFrame(self):
        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                if self.visormask[j,i] == 1:
                    self.hexes[i][j].show(BitMask32.bit(1))
                else:
                    self.hexes[i][j].hide(BitMask32.bit(1))
        render.show(BitMask32.bit(1))

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

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def takeaction(self,a1,a2,a3,speed=1):

        #action1 = move x y up down or do nothing
        if a1 == 0: self.visorparam[0] += speed
        elif a1 == 1: self.visorparam[0] -= speed
        elif a1 == 2: self.visorparam[1] += speed
        elif a1 == 3: self.visorparam[1] -= speed
        #elif a1 == 4: self.visorparam[0] += 0

        #action2 = inc h,w up down or do nothing
        if a2 == 0: self.visorparam[2] += speed
        elif a2 == 1: self.visorparam[2] -= speed
        elif a2 == 2: self.visorparam[3] += speed
        elif a2 == 3: self.visorparam[3] -= speed
        #elif a2 == 4: self.visorparam[2] += 0

        #action2 = inc h,w up down or do nothing
        if a2 == 0: self.visorparam[4] += 5 * speed * pi / 180
        elif a2 == 1: self.visorparam[4] -= 5 * speed * pi / 180
        #elif a2 == 2: self.visorparam[3] += 0

        self.incLightPos(1)
        #self.incCarPos(1)
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

    def recordScreen(self):
        base.movie(namePrefix='recording/frame',duration=5,fps=15)

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
        self.light_angle = (self.light_angle + speed) % 360

    def incCarPos(self,speed):
        self.car_x += (self.car_x + speed) % 180
        self.car.setY(sin((self.car_x)* pi / 180) * 0.1 )

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
        base.cam.setPos(x + n*x/mag,y + n*y/mag,z+n*z/mag)
        base.cam.lookAt(0,0,0)

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


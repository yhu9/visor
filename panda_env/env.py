
import random
from math import pi, sin, cos,sqrt
import numpy as np

#CUSTOM MODULES
from utils import ShadowDetector
from utils import LM_Detector

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
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
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

        # Preliminary capabilities check.
        if not base.win.getGsg().getSupportsBasicShaders():
            self.t = addTitle(
                "Shadow Demo: Video driver reports that shaders are not supported.")
            return
        if not base.win.getGsg().getSupportsDepthTexture():
            self.t = addTitle(
                "Shadow Demo: Video driver reports that depth textures are not supported.")
            return

        #SHADOW DETECTOR AND LANDMARK DETECTOR
        self.shadowfunc = ShadowDetector()
        self.lmfunc = LM_Detector()

        #base.backfaceCullingOff()
        base.setBackgroundColor(0, 0, 0, 1)
        base.camLens.setNearFar(0.1, 10000)
        base.camLens.setFov(60)

        # examine the state space
        self.width = 19
        self.height = 9
        self.state_size = (7,64,64)
        print('Size of state:', self.state_size)
        self.action_low = np.array([0,0,0,0,0])
        print('Action low:', self.action_low)
        self.action_high = np.array([self.width,self.height,self.width,self.height,3.14/2])
        print('Action high: ', self.action_high)

        #number of steps taken
        self.step_count = 0

        #initialize the scene
        self.init_scene()
        self.incrementCameraPosition(0)
        self.addControls()
        self.reset(manual_pose=1)
        self.visorpos = self.getVisorData()

        #ADD TASKS THAT RUN IF THE ENVIRONMENT IS IN A LOOP
        taskMgr.doMethodLater(.1,self.viewReward,"drawReward")
        taskMgr.doMethodLater(.1,self.spinLightTask,"spinLight")

    ##########################################################
    #CONVENIENCE FUNCTIONS
    ##########################################################
    def getInfo(self):
        self.lpos = self.getLightData()
        self.geompos = self.getVertexData()

    def getVisorData(self):
        data = [[None] * len(self.hexes[i]) for i in range(len(self.hexes))]
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[i])):
                data[i][j] = self.hexes[i][j].getPos(self.render1)
        return np.array(data)

    def getLightData(self):
        lpos = self.light.getPos(self.render1)
        return np.array([lpos[0],lpos[1],lpos[2]])

    def getVertexData(self):
        self.dennis2.flattenLight()
        def processPrim(prim,vertex,data):
            for p in range(prim.getNumPrimitives()):
                s = prim.getPrimitiveStart(p)
                e = prim.getPrimitiveEnd(p)
                for i in range(s,e):
                    vi = prim.getVertex(i)
                    vertex.setRow(vi)
                    v = vertex.getData3f()
                    data[vi] = [v[0],v[1],v[2],1]
        geomNode = self.dennis2.getChild(0).getChild(0).node()
        geoms = []
        for i in range(geomNode.getNumGeoms()):
            data = [None] * 100000
            g = geomNode.getGeom(i)
            s = geomNode.getGeomState(i)
            m = s.getAttrib(MaterialAttrib).getMaterial()
            vdata = GeomVertexReader(g.getAnimatedVertexData(True,Thread.getCurrentThread()),'vertex')
            for j in range(g.getNumPrimitives()):
                p = g.getPrimitive(j)
                processPrim(p,vdata,data)
            data = np.array(data[:data.index(None)])
            #T = LMatrix4()
            #a = geomNode.getTransform(render)
            #print(a)
            #quit()
            #T = self.dennis2.getJoints(jointName='head')[-1].getTransform()
            #T = np.array(T.getRows())
            #data = np.matmul(data,T.transpose())
            #print(data)
            #print(T)
            #quit()
            geoms.append((data[:,:3],m))

        return geoms

    def calcVisor(self,task):
        self.getInfo()
        lvec = self.lpos
        for geom,material in self.geompos:
            color = material.getDiffuse()
            if color[0] == 1 and color[1] == 0 and color[2] == 0: break

        bl = self.visorpos[0,0]
        br = self.visorpos[0,self.width-1]
        tl = self.visorpos[self.height-1,0]
        tr = self.visorpos[self.height-1,self.width-1]
        v1 = tr - tl
        v2 = bl - tl

        #GET AFFINE TRANSFORMATION MATRIX FROM WORLD COORDINATE TO VISOR MASK
        #https://stackoverflow.com/questions/22954239/given-three-points-compute-affine-transformation
        #ins = np.stack((tl[1:],bl[1:],tr[1:]))
        #out = np.array([[self.width-1,self.height-1],[self.width-1,0],[0,self.height-1]])
        #out = np.array([[0,self.height-1],[self.width-1,self.height-1],[0,0]])
        #out = np.array([[0,0],[self.width-1,0],[0,self.height-1]])
        #l = len(ins)
        #B = np.vstack([ins.transpose(),np.ones(l)])
        #D = 1.0 / np.linalg.det(B)
        #entry = lambda r,d: np.linalg.det(np.delete(np.vstack([r,B]),(d+1),axis=0))
        #M = [[(-1)**i * D * entry(R,i) for i in range(l)] for R in np.transpose(out)]
        #A, t = np.hsplit(np.array(M),[l-1])
        #t = np.transpose(t)[0]
        tl = np.hstack((tl,[1]))
        tr = np.hstack((tr,[1]))
        bl = np.hstack((bl,[1]))
        br = np.hstack((br,[1]))
        ins = np.stack((tl[1:],bl[1:],tr[1:]))
        out = np.array([[self.width-1,self.height-1],[self.width-1,0],[0,self.height-1]])
        A = np.matmul(np.linalg.inv(ins),out)

        #get intersection of eye verticies with visor plane along the light ray
        nvec = np.cross(v1,v2)  #normal to the visor
        d = np.dot(tl[:-1] + -1 * geom,nvec) / np.dot(lvec,nvec)
        #d = np.dot(tl + -1 * geom,nvec) / np.dot(lvec,nvec)
        intersection = geom + d[:,np.newaxis] * lvec
        t1 = intersection[:,1:]

        #(OPTIONAL VISUALIZATION)
        img = np.zeros((500,500,3))
        b1 = tl * 100 + 100
        b2 = tr * 100 + 100
        b3 = bl * 100 + 100
        b4 = br * 100 + 100
        vbox = np.array([b2[1:-1],b1[1:-1],b3[1:-1],b4[1:-1]])
        #vbox = np.array([b2[1:],b1[1:],b3[1:],b4[1:]])
        vbox = np.int0(vbox)
        cv2.fillConvexPoly(img,vbox,(255,255,255))
        for p in t1:
            pnt = (p * 100 + 100).astype(np.uint16)
            cv2.circle(img,(pnt[0],pnt[1]),1,(0,255,0),-1)
        cv2.imshow('img',img)
        cv2.waitKey(1)

        #get bounding box of the intersection points in 2d space collapsing the x axis
        pnts = t1[:,np.newaxis,:] * 100000 + 100000       #why 100? cuz opencv can't handle floats
        rect = cv2.minAreaRect(pnts.astype(np.int32))
        pnt,dim,r = rect
        x,y = pnt
        w,h = dim
        box = cv2.boxPoints(rect)   #tr,tl,bl,br

        #TRANSFORM INTERSECTION WORLD COORD TO VISOR MASK
        box = (box - 100000) / 100000
        #box = np.matmul(A,box.transpose()).transpose() + t
        box = np.hstack((box,np.ones((box.shape[0],1))))
        box = np.matmul(box,A)
        xpad = 2
        ypad = 2
        box = box[:,:2] + np.array([[xpad,-ypad],[-xpad,-ypad],[-xpad,ypad],[xpad,ypad]])
        box = np.rint(box)
        box[:,0] = -box[:,0] + self.width
        box[:,0] -= 3
        #box[:,1] += 4
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))
        self.shadowon()
        cv2.imshow('visormask',cv2.resize((self.visormask[:,::-1] * 255).astype(np.uint8), (self.width*10,self.height*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(1)

        return task.again

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

    #REWARD MAP GENERATION GIVEN STATE S
    def genRewardGT2(self):
        shadow_img = self.getFrame_notex()
        noshadow = (self.noshadow_img * 255).astype(np.uint8)
        shadow = (shadow_img * 255).astype(np.uint8)

        eye_mask = self.lmfunc.get_eyesGT(noshadow)     #((cx,cy),(h,w),rotation)
        shadow_mask = self.shadowfunc.get_shadowgt(shadow)
        lm = np.zeros((5,2)).astype(np.uint8)

        IOU = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(np.logical_or(eye_mask,shadow_mask))
        EYE = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(eye_mask)
        threshold = IOU + EYE

        #reward
        if threshold > 1.2:
            reward,flag = threshold, True
        else:
            reward,flag = -0.1, self.step_count == 10

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(self.visorparam,eye_mask,shadow_mask,shadow,lm,IOU,EYE,reward)

        return reward, flag, threshold

    #REWARD MAP GENERATION GIVEN STATE S
    def genRewardGT(self):
        shadow_img = self.getFrame_notex()
        noshadow = (self.noshadow_img * 255).astype(np.uint8)
        shadow = (shadow_img * 255).astype(np.uint8)

        eye_mask = self.lmfunc.get_eyesGT(noshadow)     #((cx,cy),(h,w),rotation)
        shadow_mask = self.shadowfunc.get_shadowgt(shadow)
        lm = np.zeros((5,2)).astype(np.uint8)

        IOU = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(np.logical_or(eye_mask,shadow_mask))
        EYE = np.sum(np.logical_and(eye_mask,shadow_mask)) / np.sum(eye_mask)
        thresh = IOU

        #DRAW THE SEMANTIC MASKS "OPTIONAL"
        self.drawReward(self.visorparam,eye_mask,shadow_mask,shadow,lm,IOU,EYE,thresh)

        return thresh,eye_mask,shadow_mask

    #GET THE CURRENT FRAME AS NUMPY ARRAY
    def getFrame(self):
        base.graphicsEngine.renderFrame()
        dr = self.cam.node().getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        v = memoryview(data).tolist()
        img = np.array(v,dtype=np.uint8)
        img = img.reshape((tex.getYSize(),tex.getXSize(),4))
        img = img[:,:,:3]
        img = img[::-1]
        img = img[159:601,150:503,:]        #boundry based on range of head motion
        img = cv2.resize(img,(64,64))
        return img / 255.0

    #GET THE CURRENT FRAME AS NUMPY ARRAY
    def getFrame_notex(self):
        base.graphicsEngine.renderFrame()
        dr = self.cam2.node().getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        v = memoryview(data).tolist()
        img = np.array(v,dtype=np.uint8)
        img = img.reshape((tex.getYSize(),tex.getXSize(),4))
        img = img[:,:,:3]
        img = img[::-1]
        img = img[159:601,150:503,:]        #boundry based on range of head motion
        img = cv2.resize(img,(64,64))
        return img / 255.0

    #RESET THE ENVIRONMENT
    def reset(self,manual_pose=False):
        self.step_count = 1
        if manual_pose: poseid = manual_pose
        else: poseid = random.randint(1,200)
        self.dennis.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        self.dennis2.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        self.light_angle = random.randint(-135,135)
        self.incLightPos(speed=0)                                         #PUT LIGHT IN RANDOM POSITION

        #get image without shadow
        self.shadowoff()
        self.noshadow_img = self.getFrame_notex()

        #init the visor to the same position always
        self.visorparam = [self.width//2,self.height//2,self.width//2,self.height//2,0]                              #x,y,w,h,r              #INITIAL VISOR POSITION IS ALWAYS THE SAME
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #APPLY VISOR MASK
        self.shadowon()

        #get the initial state as two copies of the first image
        base.graphicsEngine.renderFrame()
        self.prv_frame = self.getFrame()
        cur_frame = self.prv_frame
        h,w = cur_frame.shape[:2]
        frame = np.zeros((h,w,6))
        frame[:,:,:3] = self.prv_frame
        frame[:,:,3:] = cur_frame
        state = frame
        return state

    #RESET THE ENVIRONMENT
    def reset2(self,manual_pose=False):
        self.step_count = 1
        if manual_pose: poseid = manual_pose
        else: poseid = random.randint(1,200)
        self.dennis.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        self.dennis2.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        self.light_angle = random.randint(-5,5)
        self.incLightPos(speed=0)                                         #PUT LIGHT IN RANDOM POSITION

        #get image without shadow
        self.shadowoff()
        self.noshadow_img = self.getFrame_notex()

        #init the visor to the same position always
        self.visorparam = [self.width//2,self.height//2,self.width//2,self.height//2,0]                              #x,y,w,h,r              #INITIAL VISOR POSITION IS ALWAYS THE SAME
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #APPLY VISOR MASK
        self.shadowon()

        #get the initial state as two copies of the first image
        base.graphicsEngine.renderFrame()
        self.prv_frame = self.getFrame()
        cur_frame = self.prv_frame
        h,w = cur_frame.shape[:2]
        frame = np.zeros((h,w,7))

        _,goal,_ = self.genRewardGT()
        frame[:,:,:3] = self.prv_frame
        frame[:,:,3:6] = cur_frame
        frame[:,:,6] = goal.astype(np.float32)
        state = frame

        return state

    #RESET THE ENVIRONMENT
    def reset2_4(self,manual_pose=False):
        self.step_count = 1
        if manual_pose: poseid = manual_pose
        else: poseid = random.randint(1,200)
        self.dennis.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        self.dennis2.pose('head_movement',poseid)     #CHOOSE A RANDOM POSE
        #self.light_angle = random.randint(-135,135)
        #self.incLightPos(speed=0)                                         #PUT LIGHT IN RANDOM POSITION

        #get image without shadow
        self.shadowoff()
        self.noshadow_img = self.getFrame_notex()

        #init the visor to the same position always
        #self.visorparam = [self.width//2,self.height//2,self.width//2,self.height//2,0]                              #x,y,w,h,r              #INITIAL VISOR POSITION IS ALWAYS THE SAME
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #APPLY VISOR MASK
        self.shadowon()

        #get the initial state as two copies of the first image
        base.graphicsEngine.renderFrame()
        self.prv_frame = self.getFrame()
        cur_frame = self.prv_frame
        h,w = cur_frame.shape[:2]
        frame = np.zeros((h,w,7))

        _,goal,_ = self.genRewardGT()
        frame[:,:,:3] = self.prv_frame
        frame[:,:,3:6] = cur_frame
        frame[:,:,6] = goal.astype(np.float32)
        state = frame

        return self.visorparam, state

    #INITIALIZE THE 3D ENVIRONMENT
    def init_scene(self):

        #GENERATE SECOND WINDOW TO SHOW MATERIAL REGIONS
        self.win2 = base.openWindow()
        displayRegion = self.win2.makeDisplayRegion()
        self.cam2 = NodePath(Camera('cam'))
        displayRegion.setCamera(self.cam2)
        self.render2 = NodePath('render2')
        self.cam2.reparentTo(self.render2)
        self.cam2.node().getLens().setNearFar(0.1, 10000)
        self.cam2.node().getLens().setFov(60)
        self.cam2.setPos(-3.8,0.0,2.25)
        self.cam2.lookAt(-3,-0.2,2.69)
        self.dennis2 = Actor('assets/dennis.egg',{"head_movement": "assets/dennis-head_movement.egg"})
        self.dennis2.reparentTo(self.render2)
        self.dennis2.setPlayRate(0.5,'head_movement')
        self.dennis2.loop("head_movement")

        #SET MAIN DISPLAY
        display = base.win.makeDisplayRegion()
        self.cam = NodePath(Camera('cam1'))
        display.setCamera(self.cam)
        self.render1 = NodePath('render1')
        self.cam.reparentTo(self.render1)
        self.cam.node().getLens().setNearFar(0.1, 10000)
        self.cam.node().getLens().setFov(60)
        self.cam.setPos(-10.8,0.0,5.25)
        self.cam.lookAt(-3,-0.2,2.69)

        # Load the scene
        floorTex = loader.loadTexture('maps/envir-ground.jpg')
        cm = CardMaker('')
        cm.setFrame(-2, 2, -2, 2)
        floor = self.render1.attachNewNode(PandaNode("floor"))
        for y in range(12):
            for x in range(12):
                nn = floor.attachNewNode(cm.generate())
                nn.setP(-90)
                nn.setPos((x - 6) * 4, (y - 6) * 4, 0)
        floor.setTexture(floorTex)
        floor.flattenStrong()

        self.car = loader.loadModel("assets/cartex.egg")
        self.car.reparentTo(self.render1)
        self.car.set_two_sided(True)    #BEST IF I CAN SOLVE THE BIAS PROBLEM ON THE SHADER

        self.dennis = Actor('assets/dennistex.egg',{"head_movement": "assets/dennistex-head_movement.egg"})
        self.dennis.reparentTo(self.render1)
        self.dennis.setPlayRate(0.5,'head_movement')
        self.dennis.loop("head_movement")
        #CURRENTLY SHADOW QUALITY IS REDUCED DUE TO SHADOW ACNE

        self.visor, self.hexes = self.genVisor2()
        self.visor.reparentTo(self.dennis)
        self.visor.set_two_sided(True)
        self.visor.setPos(-3.75,.5,2.5)
        self.visor.setH(-90)
        self.visor.setP(90)
        self.visor.setScale(0.027,0.027,0.027)

        self.sun = DirectionalLight("Dlight")
        self.sun.color = self.sun.color * 5
        self.light = self.render1.attachNewNode(self.sun)
        self.light.node().setScene(self.render1)
        self.light.node().setShadowCaster(True)
        self.light.node().setCameraMask(BitMask32.bit(1))
        self.light.node().showFrustum()
        self.light.node().getLens().set_film_size(20)
        self.light.node().getLens().setFov(20)
        self.light.node().getLens().setNearFar(10, 50)
        self.render1.setLight(self.light)
        self.render2.setLight(self.light)

        self.alight = render.attachNewNode(AmbientLight("Ambient"))
        self.alight.node().setColor(LVector4(0.2, 0.2, 0.2, 1))
        self.render1.setLight(self.alight)
        self.render2.setLight(self.alight)

        #Important! Enable the shader generator.
        self.render1.setShaderAuto()
        self.render2.setShaderAuto()
        self.render1.show(BitMask32.bit(1))
        self.render2.show(BitMask32.bit(1))

        # default values
        self.light_angle = 0.0
        self.car_x = 0.0
        self.cameraSelection = 0
        self.lightSelection = 0
        self.visorMode = 0
        self.visormask = np.zeros((self.height,self.width))
        self.visorparam = [self.width//2,self.height//2,self.width//3,self.height//3,0]      #x,y,w,h,r
        self.light.node().hideFrustum()

    ##########################################################
    #Controls
    ##########################################################
    #SOME CONTROL SEQUENCES
    def addControls(self):
        #self.inst_p = addInstructions(0.06, 'up/down arrow: zoom in/out')
        #self.inst_x = addInstructions(0.12, 'Left/Right Arrow : switch camera angles')
        #addInstructions(0.18, 'a : put sun on face')
        #addInstructions(0.24, 's : toggleSun')
        #addInstructions(0.30, 'd : toggle visor')
        #addInstructions(0.36, 'v: View the Depth-Texture results')
        #addInstructions(0.42, 'tab : view buffer')
        self.anim_flag = False
        self.accept('escape', self.exit)
        #self.accept("a", self.putSunOnFace)
        self.accept("d", self.toggleVisor,[1])
        self.accept("r", self.recordScreen)
        self.accept("arrow_left", self.incrementCameraPosition, [-1])
        self.accept("arrow_right", self.incrementCameraPosition, [1])
        self.accept("f12",base.screenshot,['recording/snapshot'])
        self.accept("tab", base.bufferViewer.toggleEnable)
        self.accept("p", self.addAnimation)

    def exit(self):
        quit()

    def addAnimation(self):
        self.anim_flag = not self.anim_flag
        if self.anim_flag:
            self.dennis.setPlayRate(0.5,'head_movement')
            self.dennis.loop('head_movement')
            self.dennis2.setPlayRate(0.5,'head_movement')
            self.dennis2.loop('head_movement')
        else:
            self.dennis.stop('head_movement')
            self.dennis2.stop('head_movement')


    def putSunOnFace(self):
        self.lightSelection = 0
        self.light.setPos(-15,0,3)
        self.light.lookAt(0,0,0)

    def toggleVisor(self,n):
        self.visorMode = (self.visorMode + n) % 3
        if (self.visorMode == 0):
            taskMgr.remove('random controller')
        if (self.visorMode == 1):
            taskMgr.remove("random controller")
            taskMgr.doMethodLater(0.1,self.calcVisor,"calcVisor")
            for row in self.hexes:
                for h in row:
                    h.show()
            render.show()
        if (self.visorMode == 2):
            taskMgr.remove('calcVisor')
            taskMgr.doMethodLater(0.5,self.randomVisor,'random controller')

    def recordScreen(self):
        base.movie(namePrefix='recording/frame',duration=5,fps=15)

    ##########################################################
    #TASKS
    ##########################################################
    def viewReward(self,task):
        self.genRewardGT()
        return Task.again

    def randomVisor(self,task):

        for _ in range(300):
            i = random.randint(0,len(self.hexes)-1)
            j = random.randint(0,len(self.hexes[-1]) - 1)
            self.hexes[i][j].hide()

        for _ in range(300):
            i = random.randint(0,len(self.hexes)-1)
            j = random.randint(0,len(self.hexes[-1]) - 1)
            self.hexes[i][j].show()

        render.show()
        return Task.again


    def shadowoff(self):
        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                self.hexes[i][j].hide()

    def shadowon(self):
        #APPLY VISOR MASK
        for i in range(len(self.hexes)):
            for j in range(len(self.hexes[-1])):
                if self.visormask[i,j] == 1:
                    self.hexes[i][j].show()
                else:
                    self.hexes[i][j].hide()

    def getstate(self,prv_frame,cur_frame):
        h,w = cur_frame.shape[:2]
        d = 6
        frame = np.zeros((h,w,d))
        frame[:,:,:3] = prv_frame
        frame[:,:,3:] = cur_frame
        state = frame
        return state

    #get state
    def getstate2(self,prv_frame,cur_frame,goal):
        h,w = cur_frame.shape[:2]
        d = 7
        frame = np.zeros((h,w,d))
        frame[:,:,:3] = prv_frame
        frame[:,:,3:6] = cur_frame
        frame[:,:,6] = goal.astype(np.float32)
        state = frame
        return state

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def step2_4(self,actions,speed=1):
        self.step_count += 1

        a1,a2,a3 = [a for a in actions]
        visor = self.visorparam

        #action1 = move x y up down
        if a1 == 0: self.visorparam[0] += speed
        elif a1 == 1: self.visorparam[0] -= speed
        elif a1 == 2: self.visorparam[1] += speed
        elif a1 == 3: self.visorparam[1] -= speed

        #action2 = inc h,w up down
        if a2 == 0: self.visorparam[2] += speed
        elif a2 == 1: self.visorparam[2] -= speed
        elif a2 == 2: self.visorparam[3] += speed
        elif a2 == 3: self.visorparam[3] -= speed

        #action3 = inc theta up down
        if a2 == 0: self.visorparam[4] += 5 * speed * pi / 180
        elif a2 == 1: self.visorparam[4] -= 5 * speed * pi / 180

        #get image with shadow after action
        self.incLightPos()
        self.visorparam[0] = min(max(0,self.visorparam[0]),self.width-1)
        self.visorparam[1] = min(max(0,self.visorparam[1]),self.height-1)
        self.visorparam[2] = min(max(0,self.visorparam[2]),self.width-1)
        self.visorparam[3] = min(max(0,self.visorparam[3]),self.height-1)
        self.visorparam[4] = min(max(-pi,self.visorparam[4]),pi)
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])     #PADDED HEIGHT AND WIDTH OF 15PIXELS
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        self.shadowon()
        cur_frame = self.getFrame()
        cv2.imshow('prv',(self.prv_frame * 255).astype(np.uint8))
        cv2.imshow('cur',(cur_frame*255).astype(np.uint8))
        cv2.imshow('visormask',cv2.resize((self.visormask[:,::-1]* 255).astype(np.uint8), (self.width*10,self.height*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(1)

        #get next state and reward
        reward,eye_mask,shadow_mask = self.genRewardGT()
        done = self.step_count >= 10 or reward > 0.25

        #set the next state
        next_state = self.getstate2(self.prv_frame,cur_frame,shadow_mask)
        self.prv_frame = cur_frame.copy()

        return visor,next_state,reward,done

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def step_1_6(self,actions,speed=1):
        self.step_count += 1

        a1,a2,a3 = [a for a in actions]

        #action1 = move x y up down
        if a1 == 0: self.visorparam[0] += speed
        elif a1 == 1: self.visorparam[0] -= speed
        elif a1 == 2: self.visorparam[1] += speed
        elif a1 == 3: self.visorparam[1] -= speed

        #action2 = inc h,w up down
        if a2 == 0: self.visorparam[2] += speed
        elif a2 == 1: self.visorparam[2] -= speed
        elif a2 == 2: self.visorparam[3] += speed
        elif a2 == 3: self.visorparam[3] -= speed

        #action3 = inc theta up down
        if a2 == 0: self.visorparam[4] += 5 * speed * pi / 180
        elif a2 == 1: self.visorparam[4] -= 5 * speed * pi / 180

        #get image with shadow after action
        #self.incLightPos()
        self.visorparam[0] = min(max(0,self.visorparam[0]),self.width-1)
        self.visorparam[1] = min(max(0,self.visorparam[1]),self.height-1)
        self.visorparam[2] = min(max(0,self.visorparam[2]),self.width-1)
        self.visorparam[3] = min(max(0,self.visorparam[3]),self.height-1)
        self.visorparam[4] = min(max(-pi,self.visorparam[4]),pi)
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])     #PADDED HEIGHT AND WIDTH OF 15PIXELS
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        self.shadowon()
        cur_frame = self.getFrame()
        cv2.imshow('prv',(self.prv_frame * 255).astype(np.uint8))
        cv2.imshow('cur',(cur_frame*255).astype(np.uint8))
        cv2.imshow('visormask',cv2.resize((self.visormask[:,::-1]* 255).astype(np.uint8), (self.width*10,self.height*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(1)

        #get next state and reward
        reward,eye_mask,shadow_mask = self.genRewardGT()
        done = self.step_count >= 10 or reward > 0.25

        #set the next state
        next_state = self.getstate2(self.prv_frame,cur_frame,shadow_mask)
        self.prv_frame = cur_frame.copy()

        return next_state,reward,done

    #take a possible of 10 actions to move x,y,w,h,r up or down
    #and update the visor mask accordingly
    def step(self,actions,speed=1):
        self.step_count += 1

        actions = (actions + 1) * (self.action_low + self.action_high) / 2
        self.visorparam = actions

        #get image with shadow after action
        #self.incLightPos()
        self.visorparam[0] = min(max(0,self.visorparam[0]),self.width-1)
        self.visorparam[1] = min(max(0,self.visorparam[1]),self.height-1)
        self.visorparam[2] = min(max(0,self.visorparam[2]),self.width-1)
        self.visorparam[3] = min(max(0,self.visorparam[3]),self.height-1)
        self.visorparam[4] = min(max(-pi,self.visorparam[4]),pi)
        rot_rect = ((self.visorparam[0],self.visorparam[1]),(self.visorparam[2],self.visorparam[3]),self.visorparam[4])
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        self.visormask *= 0
        cv2.fillConvexPoly(self.visormask,box,(1))

        #display the visor and show some data
        self.shadowon()
        cur_frame = self.getFrame()
        cv2.imshow('prv',(self.prv_frame * 255).astype(np.uint8))
        cv2.imshow('cur',(cur_frame*255).astype(np.uint8))
        cv2.imshow('visormask',cv2.resize((self.visormask[:,::-1] * 255).astype(np.uint8), (self.width*10,self.height*10), interpolation = cv2.INTER_LINEAR))
        cv2.waitKey(1)

        #get next state and reward and stopping flag
        reward,eye_mask,shadow_mask = self.genRewardGT()
        done = self.step_count >= 5 or reward > 0.20

        #set the next state
        next_state = self.getstate2(self.prv_frame,cur_frame,shadow_mask)
        self.prv_frame = cur_frame.copy()
        return next_state,reward,done

    def spinLightTask(self,task):
        angleDegrees = (self.light_angle - 80)
        angleRadians = angleDegrees * (pi / 180.0)
        self.light.setPos(-15.0,2+3.0 * cos(angleRadians),2.3 + 0.6 * cos(angleRadians * 4.0))
        self.light.lookAt(0,0,0)
        self.light_angle += 5
        return task.again

    def incLightPos(self,speed=2):
        angleRadians = self.light_angle * (pi / 180.0)
        self.light.setPos(-15.0,2 + 3.0 * cos(angleRadians),2.3 + 0.6 * cos(angleRadians * 4.0))
        self.light.lookAt(0,0,0)
        self.light_angle += speed

    def incCarPos(self,speed):
        self.car_x += (self.car_x + speed) % 180
        self.car.setY(sin((self.car_x)* pi / 180) * 0.1 )

    def incrementCameraPosition(self,n):
        self.cameraSelection = (self.cameraSelection + n) % 3
        if (self.cameraSelection == 1):
            self.cam.setPos(-7,0,3)
            self.cam.lookAt(0,0,0)

        if (self.cameraSelection == 0):
            #self.camLens.setNearFar(0,10)
            self.cam.setPos(-3.8,0.0,2.25)
            self.cam.lookAt(-3,-0.2,2.69)

        if (self.cameraSelection == 2):
            self.cam.setPos(self.light.getPos())
            self.cam.lookAt(0,0,0)

    def genVisor2(self):
        visor = self.render1.attach_new_node("visor")
        objects = [[None] * self.width for i in range(self.height)]
        r = 1.0
        y = 0.866025
        x = 0.5
        offsetx = (1.55) / 2.00
        offsety = (sqrt(1 - (offsetx * offsetx)) + 1) / 2.00
        #x,y = 0,0
        for i in range(0,self.height):
            for j in range(0,self.width):
                cx = offsety + (2*y*j) + (y*(i%2))
                cy = offsetx + ((r + x) * i)
                objects[i][j] = loader.loadModel("assets/hex/hexagon.egg")
                objects[i][j].reparentTo(visor)
                objects[i][j].setPos(cx, cy,5)
                objects[i][j].setScale(1.0,1.0,1.0)
                objects[i][j].setAlphaScale(0.01)
        return visor,objects

#################################################################################
#################################################################################
#################################################################################
#################################################################################

if __name__ == '__main__':
    w = World()
    base.run()


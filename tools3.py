import sys
import os
import time

import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
#from scipy.ndimage import imread
#import scipy
from imutils import face_utils

import dlib
from faced import FaceDetector
from faced.utils import annotate_image

class ShadowDetector():

    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(varThreshold=500)

    #single image shadow detector using ycrcb color space
    def get_shadow(self,rgb):

        illum_mask = np.all(rgb != [0,0,0],axis=-1)     #grab background
        ycrcb = cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)   #cvt color space to ycrcb
        ycrcb = cv2.GaussianBlur(ycrcb,(5,5),3)         #blur the image
        y_mean = np.mean(ycrcb[illum_mask][:,0])        #grab the mean
        #y_std = np.std(ycrcb[illum_mask][:,0])         #grab the std
        mask = ycrcb[:,:,0] < y_mean                    #ycrcb[:,:,0] < y_mean - y_std / a
        mask = mask * illum_mask                        #remove background

        return mask

    #video shadow detector using background subtraction
    def get_shadow2(self,rgb):
        mog_mask = self.mog.apply(rgb)
        return mog_mask == 127

        #plt.imshow(mog_mask)
        #plt.show()

class LM_Detector():
    def __init__(self):

        self.fa3 = cv2.dnn.readNetFromCaffe("./models/deploy.prototxt.txt","./models/res10_300x300_ssd_iter_140000.caffemodel")
        self.fa2 = dlib.get_frontal_face_detector()
        self.fa = FaceDetector()
        self.cnn_dlib = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
        self.predictor = dlib.shape_predictor("./models/shape_predictor.dat")
        self.predictor5 = dlib.shape_predictor("./models/shape_predictor5.dat")
        self.eye_cascade = cv2.CascadeClassifier('./models/haarcascade_eye.xml')
        self.thresh = 0.5

    #DLIB SIMPLE LANDMARK DETECTION
    #1. uses naive face detection
    #2. uses naive resnet 10 lm detection
    def lm2(self,rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        bbox = self.fa.predict(gray,0)
        return [face_utils.shape_to_np(self.predictor(gray,box)) for box in bbox]

    #DLIB SIMPLE LANDMARK DETECTION
    #1. uses naive face detection
    #2. uses naive resnet 10 lm detection
    def lm3(self,rgb,save=False):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        st_time = time.time()
        bbox = self.fa2(gray,0)
        print(time.time() - st_time)

        #FOR VISUALIZATION ONLY
        if save and len(bbox) > 0:
            #box = self.get_driver(bbox)
            for box in bbox:
                cv2.rectangle(rgb,(box.left(),box.top()),(box.right(),box.bottom()),[255,0,0],2)

        return [face_utils.shape_to_np(self.predictor(gray,box)) for box in bbox]

    #DLIB SIMPLE LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def lm4(self,rgb,save=False):
        if rgb.shape[-1] == 4:
            rgb = rgb[:,:,:3].astype(np.uint8)
        bbox = self.fa.predict(rgb,self.thresh)

        #FOR VISUALIZATION ONLY
        if save and len(bbox) > 0:
            #box = self.get_driver(bbox)
            #rgb = annotate_image(rgb,bbox)
            for box in bbox:
                l = box[1] - box[2]//2
                t = box[0] - box[3]//2
                r = box[1] + box[2]//2
                b = box[0] + box[3]//2
                cv2.rectangle(rgb,(l,t),(r,b),[255,0,0],2)

        return [face_utils.shape_to_np(self.predictor(rgb,dlib.rectangle(box[0] - box[2]//2,box[1] - box[3]//2,box[0] + box[2]//2,box[1] + box[3]//2))) for box in bbox]

    #DLIB 5 point LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def lm5(self,rgb,save=False):
        bbox = self.fa.predict(rgb,self.thresh)

        #FOR VISUALIZATION ONLY
        if save and len(bbox) > 1:
            box = bbox[1]
            l = box[0] - box[2]//2
            t = box[1] - box[3]//2
            r = box[0] + box[2]//2
            b = box[1] + box[3]//2
            cv2.rectangle(rgb,(l,t),(r,b),[255,0,0],2)

        return [face_utils.shape_to_np(self.predictor5(rgb,dlib.rectangle(box[0] - box[2]//2,box[1] - box[3]//2,box[0] + box[2]//2,box[1] + box[3]//2))) for box in bbox]

    #CV2 HAAR CASCADE EYE DETECTOR (the popular one)
    def lm6(self, rgb,save=False):
        bbox = self.fa.predict(rgb,self.thresh)
        if len(bbox) > 0:
            box = bbox[0]
            l = box[0] - box[2]//2
            t = box[1] - box[3]//2
            r = box[0] + box[2]//2
            b = box[1] + box[3]//2
            roi = rgb[t:b,l:r]
            gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray)

            if save:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            return eyes

    def lm7(self,rgb,save=False):

        st_time = time.time()
        dets = self.cnn_dlib(rgb,0)
        print(time.time() - st_time)
        if save and len(dets) > 0:
            for i, d in enumerate(dets):

                cv2.rectangle(rgb,(d.rect.left(),d.rect.top()),(d.rect.right(),d.rect.bottom()),[255,0,0],2)

        return [face_utils.shape_to_np(self.predictor5(rgb,d.rect)) for d in dets]

    #DLIB SIMPLE LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def lm8(self,rgb,pad=20):
        h,w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb,(300,300)),1.0,(300,300),(103.93,116.77,123.68))
        self.fa3.setInput(blob)
        detections = self.fa3.forward()

        #get driver bounding box based on rightmost
        rightmost = -1
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            if confidence > 0.7 and box[0] > rightmost:
                rightmost = box[0]
                box = box.astype("int")
                bbox = dlib.rectangle(box[0],box[1],box[2],box[3])
                #(startX,startY,endX,endY) = box.astype("int")
                #rgb = cv2.rectangle(rgb,(startX,startY),(endX,endY),(0,0,255),2)
        #cv2.imshow("image",rgb)
        #cv2.waitKey(0)
        #quit()
        if rightmost == -1: return
        lm = face_utils.shape_to_np(self.predictor5(rgb,bbox))
        lm[:,0] = lm[:,0] - bbox.left()
        lm[:,1] = lm[:,1] - bbox.top()
        rgb = rgb[bbox.top():bbox.bottom(),bbox.left():bbox.right()]
        return rgb,lm

    def get_driver(self,boxes): #get bounding box for the driver only
        rightmost = 0
        index = -1
        if isinstance(boxes,dlib.rectangles):
            for i, b in enumerate(boxes):
                if b.right() > rightmost: rightmost = b.right(); index=i
            return boxes[index]

        if len(boxes) > 0:
            for i,b in enumerate(boxes):
                r = b[0] + b[2] // 2
                if r > rightmost: rightmost = r; index=i
            return boxes[index]

        return

    def plot_lm(self,rgb,lm):
        for x,y in lm:
            cv2.circle(rgb,(x,y),3,(255,0,0),-1)

    def get_face(self,rgb,lm,pad=50):

        t = np.amin(lm[:,1])
        l = np.amin(lm[:,0])
        b = np.amax(lm[:,1])
        r = np.amax(lm[:,0])
        h,w = rgb.shape[:2]

        return rgb[max(t-pad,0):min(b+pad,h-1),max(l-pad,0):min(r+pad,w-1)]

######################################################################################
######################################################################################
######################################################################################
#UNIT TESTS

def test_getface():
    DIR_IN = "/home/mhu/DATA/visor/2019-05-07"
    lm_detector = LM_Detector()

    if not os.path.exists('sample_faces'):
        os.mkdir('sample_faces')

    for f in os.listdir(DIR_IN):
        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)

        img,preds = lm_detector.lm8(rgb)
        plt.imsave(os.path.join('sample_faces',f), img)


def test_lm(DIR_IN="/home/mhu/DATA/visor/2019-05-07"):

    lm_detector = LM_Detector()
    total = 0.0
    fneg = 0
    count = 0
    for f in os.listdir(DIR_IN):
        count += 1

        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)

        st = time.time()
        #preds = lm_detector.lm2(rgb)
        preds = lm_detector.lm3(rgb,save=True)
        #preds = lm_detector.lm4(rgb,save=True)
        #preds = lm_detector.lm5(rgb,save=True)     #MY CURRENT FAVORITE
        #preds = lm_detector.lm6(rgb,save=True)
        #preds = lm_detector.lm7(rgb,save=False)
        exec_time = time.time() - st
        #print(exec_time,count)
        total = total + exec_time

        if len(preds) == 0: fneg += 1
        if preds:   #if a face was found
            for face_lm in preds:
                lm_detector.plot_lm(rgb,face_lm)
                cv2.imshow('rgb',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                cv2.waitKey(100)
        else: print("ERROR NO FACE FOUND")
        #plt.imsave(os.path.basename(fin) + '.png',rgb,format='png')

    print("%i / %i faced detected" % (count - fneg,count))
    print(total)
    print("avg time: %.8f" % (total / count))

def test_lm2():
    DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    vid = imageio.get_reader(DIR_IN,'ffmpeg')

    lm_detector = LM_Detector()

    total = 0.0
    fneg = 0
    count = 0

    for i,rgb in enumerate(vid):
        count += 1

        st = time.time()
        #preds = lm_detector.lm2(rgb)
        #preds = lm_detector.lm3(rgb,save=True)
        #preds = lm_detector.lm4(rgb,save=True)
        #preds = lm_detector.lm5(rgb,save=True)     #MY CURRENT FAVORITE
        #preds = lm_detector.lm6(rgb,save=True)
        exec_time = time.time() - st
        total = total + exec_time

        if len(preds) == 0: fneg += 1

        if preds:   #if a face was found
            for face_lm in preds:
                lm_detector.plot_lm(rgb,face_lm)
                cv2.imshow('rgb',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                cv2.waitKey(10)
        else: print("ERROR NO FACE FOUND")
        #plt.imsave(os.path.basename(fin) + '.png',rgb,format='png')

    print("%i / %i faced detected" % (count - fneg,count))
    print(total)
    print("avg time: %.8f" % (total / count))

def test_shadow():
    #DIR_IN = "/home/mhu/DATA/visor/2019-05-07"
    DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    vid = imageio.get_reader(DIR_IN,'ffmpeg')

    lm_detector = LM_Detector()
    detector = ShadowDetector()

    for i,v in enumerate(vid):

        preds = lm_detector.lm4(v)
        rgb = lm_detector.get_face(v,preds[-1])
        mask = detector.get_shadow(rgb)
        lm_detector.plot_lm(v,preds[-1])

        img = np.full(mask.shape,255)
        img *= mask
        cv2.imshow('color',cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
        cv2.imshow('img',img.astype(np.uint8))
        cv2.waitKey(50)

def test_shadow2():
    #DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    DIR_IN = "panda_env/recording/out.avi"
    vid = imageio.get_reader(DIR_IN,'ffmpeg')
    detector = ShadowDetector()
    lm_detector = LM_Detector()

    for i,v in enumerate(vid):
        preds = lm_detector.lm4(v)
        rgb = lm_detector.get_face(v,preds[-1])
        mask = detector.get_shadow2(v)
        #lm_detector.plot_lm(v,preds[-1])

        img = np.full(mask.shape,255)
        img *= mask
        cv2.imshow('rgb',cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
        cv2.imshow('img',img.astype(np.uint8))
        cv2.waitKey(10)

def test_shadow3():
    DIR_IN = "panda_env/recording"
    detector = ShadowDetector()
    lm_detector = LM_Detector()

    for f in os.listdir(DIR_IN):
        v = plt.imread(os.path.join(DIR_IN,f))
        preds = lm_detector.lm4(v,save=True)
        rgb = lm_detector.get_face(v,preds[-1])
        mask = detector.get_shadow(rgb)
        lm_detector.plot_lm(rgb,preds[-1])

        img = np.full(mask.shape,255)
        img *= mask
        cv2.imshow('rgb',cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
        cv2.imshow('img',img.astype(np.uint8))
        cv2.waitKey(10)

def test_shadow4():
    #DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    DIR_IN = "panda_env/recording/out.avi"
    vid = imageio.get_reader(DIR_IN,'ffmpeg')
    detector = ShadowDetector()
    lm_detector = LM_Detector()

    for i,v in enumerate(vid):
        preds = lm_detector.lm4(v)
        rgb = lm_detector.get_face(v,preds[-1])
        mask = detector.get_shadow(rgb)
        lm_detector.plot_lm(v,preds[-1])

        img = np.full(mask.shape,255)
        img *= mask
        cv2.imshow('rgb',cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
        cv2.imshow('img',img.astype(np.uint8))
        cv2.waitKey(50)

def get_faces():
    DIR_IN = "/home/mhu/DATA/visor/2019-05-07"
    lm_detector = LM_Detector()

    for f in os.listdir(DIR_IN):
        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)

        st = time.time()
        preds = lm_detector.lm7(rgb,save=False)
        exec_time = time.time() - st
        print(exec_time,count)
        #total = total + exec_time

        if len(preds) == 0: fneg += 1
        if preds:   #if a face was found
            for face_lm in preds:
                lm_detector.plot_lm(rgb,face_lm)
                cv2.imshow('rgb',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                cv2.waitKey(100)
        else: print("ERROR NO FACE FOUND")
        #plt.imsave(os.path.basename(fin) + '.png',rgb,format='png')

def test_shadow5():
    DIR_IN = "./sample_faces"
    #DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    #vid = imageio.get_reader(DIR_IN,'ffmpeg')

    detector = ShadowDetector()

    total_time = 0.0
    for f in os.listdir(DIR_IN):
        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)

        st_time = time.time()
        mask = detector.get_shadow(rgb)
        total_time += time.time() - st_time
        img = np.full(mask.shape,255)
        img *= mask
        #plt.imsave(f,img,cmap='gray')

    print('avg time: ' + str(total_time / 26))

######################################################################################
######################################################################################
######################################################################################
#testing site via main

if __name__ == '__main__':

    #test_lm(DIR_IN = "panda_env/recording")
    #test_shadow(DIR_IN="panda_env/recording")
    #test_shadow2()
    #test_shadow3()
    #test_shadow4()
    test_shadow5()
    #test_lm()
    #test_lm2()
    #test_lm()
    #test_getface()
    #get_faces()


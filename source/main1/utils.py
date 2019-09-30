"""
    Author: Masa Hu
    Email: huynshen@msu.edu

    tools3.py is a simple overview of the different techniques we investigated on 2d/3d landmark localization, face detection,
    and shadow detection. No AI solution was investigated on shadow detection as all state of the art approaches involving AI
    required GPU capabilities to run efficiently, while still performing sub-optimally.

    you can run tools3.py as main to see a small demo
"""

#Native library imports
import sys
import os
import time

#OPEN SOURCE IMPORTS
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
from imutils import face_utils
import dlib
import face_alignment
from faced import FaceDetector
from faced.utils import annotate_image

#################################################################################################
#################################################################################################
#################################################################################################
#SHADOW DETECTION CLASS
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

#################################################################################################
#################################################################################################
#################################################################################################
#FACE DETECTION CLASS
class FADetector():
    def __init__(self):
        self.fa = FaceDetector()
        self.fa2 = dlib.get_frontal_face_detector()
        self.fa3 = cv2.dnn.readNetFromCaffe("./models/deploy.prototxt.txt","./models/res10_300x300_ssd_iter_140000.caffemodel")
        self.fa4 = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
        self.thresh = 0.5

    #DLIB SIMPLE LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def cv2dnn_facedetection(self,rgb,pad=20):
        h,w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb,(300,300)),1.0,(300,300),(103.93,116.77,123.68))
        self.fa3.setInput(blob)
        detections = self.fa3.forward()

        #get driver bounding box based on rightmost position
        rightmost = -1
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            if confidence > 0.7 and box[0] > rightmost:
                rightmost = box[0]
                box = box.astype("int")
                bbox = dlib.rectangle(box[0],box[1],box[2],box[3])
        if rightmost == -1: return

        return rgb[bbox.top():bbox.bottom(),bbox.left():bbox.right()]

    #YOLO FACE DETECTION FROM https://github.com/iitzco/faced
    def yolo_facedetection(self,rgb):
        bbox = self.fa.predict(rgb,self.thresh)
        box = bbox[0]
        l = box[1] - box[2] // 2
        t = box[0] - box[3] // 2
        r = box[1] + box[2] // 2
        b = box[0] + box[3] // 2
        return rgb[t:b,l:r]

    #CNN FACE DETECTION FROM DLIB
    def dlibcnn_facedetection(self,rgb,save=False):
        dets = self.fa4(rgb,0)
        d = dets[0]
        return rgb[d.rect.top():d.rect.bottom(),d.rect.left():d.rect.right()]

#################################################################################################
#################################################################################################
#################################################################################################
#LANDMARK DETECTION CLASS
class LM_Detector():
    def __init__(self):
        self.bulat = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,device='cpu')
        self.predictor = dlib.shape_predictor("../../models/shape_predictor.dat")
        self.predictor5 = dlib.shape_predictor("../../models/shape_predictor5.dat")
        self.eye_cascade = cv2.CascadeClassifier('../../models/haarcascade_eye.xml')

    #DLIB SIMPLE LANDMARK DETECTION
    #1. uses naive resnet 10 lm detection
    def dlibresnet_lmdetector(self,rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        box = dlib.rectangle(0,0,gray.shape[1] - 1,gray.shape[0] - 1)
        preds = face_utils.shape_to_np(self.predictor(gray,box))
        return preds

    #DLIB 5 point LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def dlib5pt_lmdetector(self,rgb):
        box = dlib.rectangle(0,0,rgb.shape[1] - 1,rgb.shape[0] - 1)
        preds = face_utils.shape_to_np(self.predictor5(rgb,box))
        return preds

    #CV2 HAAR CASCADE EYE DETECTOR (the popular one)
    def cv2haarcascade_lmdetector(self, rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        return eyes

    #ADRIAN BULAT LANDMARK DETECTION
    def bulat_lmdetector(self,rgb):
        preds = self.bulat.get_landmarks(rgb)
        return preds[0]

    #HELPER FUNCTION TO VISUALIZE LANDMARK PREDICTIONS ON RGB IMAGE
    def plot_lm(self,rgb,lm):
        if lm.shape[1] == 4:
            for x,y,w,h in lm:
                cv2.circle(rgb,(x+w//2,y+h//2),3,(255,0,0),-1)
        elif lm.shape[1] == 2:
            for x,y in lm:
                cv2.circle(rgb,(x,y),3,(255,0,0),-1)

######################################################################################
######################################################################################
######################################################################################
#EVALUATION FUNCTIONS FOR EACH METHOD

#OPENCV'S FACE DETECTION MODULE FROM ITS DNN LIBRARY
def test_getface(mode=''):
    DIR_IN = "/home/mhu/DATA/visor/2019-05-07"
    fa_detector = FADetector()
    times = []
    for f in os.listdir(DIR_IN):
        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)
        cv2.imshow("In Img", cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

        #PERFORM FACE DETECTION
        st_time = time.time()
        if mode == 'cv2dnn':
            img = fa_detector.cv2dnn_facedetection(rgb)
        elif mode == 'yolo-cpu':
            img = fa_detector.yolo_facedetection(rgb)
        elif mode == 'dlib-cnn':
            img = fa_detector.dlibcnn_facedetection(rgb)
        times.append(time.time() - st_time)

        cv2.imshow('Out Img',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)
    print("AVG TIME = " + str(np.mean(times)))

#TEST THE DIFFERENT LANDMARK DETECTORS
def test_lm(DIR_IN="sample_faces",mode=''):
    lm = LM_Detector()
    total = 0.0

    for i,f in enumerate(os.listdir(DIR_IN)):
        fin = os.path.join(DIR_IN,f)
        rgb = plt.imread(fin)
        cv2.imshow("In Img", cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

        #LANDMARK DETECTION
        st = time.time()
        if mode == 'resnet':
            preds = lm.dlibresnet_lmdetector(rgb)
        elif mode == 'resnet5pt':
            preds = lm.dlib5pt_lmdetector(rgb)
        elif mode == 'haarcascade':
            preds = lm.cv2haarcascade_lmdetector(rgb)     #MY CURRENT FAVORITE
        elif mode == 'bulat':
            preds = lm.bulat_lmdetector(rgb)
        else: break
        exec_time = time.time() - st
        total = total + exec_time

        #PLOT THE PREDICTED LANDMARKS FOR VISUALIZATION
        if len(preds) > 0:
            lm.plot_lm(rgb,preds)
            cv2.imshow('Out Img',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)
        else: print("NO EYES FOUND")
        #plt.imsave(os.path.basename(fin) + '.png',rgb,format='png')
    print("avg time: %.8f" % (total / (i+1)))

#TEST SHADOW DETECTION ON MORE REALISTIC VIDEO
def test_shadow():
    #DIR_IN = "/home/mhu/DATA/visor/2019-05-07"
    DIR_IN = "/home/mhu/DATA/driver_vid/clips/output10.mp4"
    vid = imageio.get_reader(DIR_IN,'ffmpeg')

    fa_detector = FADetector()
    detector = ShadowDetector()

    for i,v in enumerate(vid):
        rgb = fa_detector.cv2dnn_facedetection(v)
        if not type(rgb) == type(None):
            mask = detector.get_shadow(rgb)
            img = np.full(mask.shape,255)
            img *= mask
            cv2.imshow('IN IMG',cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
            cv2.imshow('IN FACE', cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imshow('OUT MASK',img.astype(np.uint8))
            cv2.waitKey(10)

#TEST SHADOW DETECTION
def test_shadow3():
    DIR_IN = "./sample_faces"
    detector = ShadowDetector()
    fa_detector = FADetector()
    times = []

    for f in os.listdir(DIR_IN):
        rgb = plt.imread(os.path.join(DIR_IN,f))
        cv2.imshow('IN IMG',cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))

        st_time = time.time()
        mask = detector.get_shadow(rgb)
        times.append(time.time() - st_time)

        img = np.full(mask.shape,255)
        img *= mask
        cv2.imshow('OUT IMG',img.astype(np.uint8))
        cv2.waitKey(100)
    print("AVG TIME = " + str(np.mean(times)))

######################################################################################
######################################################################################
######################################################################################
#testing site via main

if __name__ == '__main__':

    print("\r\r\r")
    print("   #################################")
    print("########### Shadow Detection #############")
    print("   #################################")
    print("############## Thresholding  On Ideal Data ##############")
    test_shadow3()
    print("############## Thresholding on Non-Ideal Data ##############")
    test_shadow()

    #quit()
    print("\r\r\r")
    print("   #################################")
    print("########### Face Detection #############")
    print("   #################################")
    print("############## MODEL CV2 DNN ##############")
    test_getface(mode='cv2dnn')
    print("############## MODEL YOLO CPU ##############")
    test_getface(mode='yolo-cpu')
    print("############## MODEL DLIB CNN ##############")
    test_getface(mode='dlib-cnn')

    #quit()
    print("\r\r\r")
    print("     #################################")
    print("########## Landmark Detection ############")
    print("     #################################")
    print("############## MODEL RESNET ##############")
    test_lm(mode='resnet')
    print("############## MODEL RESNET 5 PT ##############")
    test_lm(mode='resnet5pt')
    print("############## MODEL HAAR CASCADE FILTERS ##############")
    test_lm(mode='haarcascade')
    print("############## MODEL ADRIAN BULAT ##############")
    test_lm(mode='bulat')


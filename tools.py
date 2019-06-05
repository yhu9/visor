import sys
import os
import time

import cv2
import face_alignment
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from imutils import face_utils
import dlib

class LM_Detector():
    def __init__(self):

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,device='cpu')
        self.fa2 = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./models/shape_predictor.dat")
        self.predictor5 = dlib.shape_predictor("./models/shape_predictor5.dat")

    #ADRIAN BULAT STATE OF THE ART FACE/LANDMARK DETECTION
    def lm1(self,rgb):
        return self.fa.get_landmarks(rgb)

    #DLIB SIMPLE LANDMARK DETECTION
    #1. uses naive face detection
    #2. uses naive resnet 10 lm detection
    def lm2(self,rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        bbox = self.fa2(gray,0)
        return [face_utils.shape_to_np(self.predictor(gray,box)) for box in bbox]

    #DLIB SIMPLE LANDMARK DETECTION
    #1. uses naive face detection
    #2. uses naive resnet 10 lm detection
    def lm3(self,rgb):
        gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        bbox = self.fa2(gray,0)
        return [face_utils.shape_to_np(self.predictor5(gray,box)) for box in bbox]

    def plot_lm(self,rgb,lm):
        for x,y in lm:
            cv2.circle(rgb,(x,y),3,(255,0,0),-1)

if __name__ == '__main__':

    DIR_IN = "/home/mhu/DATA/visor/2019-05-07"

    lm_detector = LM_Detector()
    total = 0.0
    fneg = 0
    for f in os.listdir(DIR_IN):

        fin = os.path.join(DIR_IN,f)
        rgb = imread(fin)

        st = time.time()
        preds = lm_detector.lm1(rgb)
        #preds = lm_detector.lm2(rgb)
        #preds = lm_detector.lm3(rgb)
        exec_time = time.time() - st
        total = total + exec_time
        print(exec_time)

        if not preds: fneg += 1
        #if preds:   #if a face was found
        #    for face_lm in preds:
        #        lm_detector.plot_lm(rgb,face_lm)
        #plt.imsave(os.path.basename(fin) + '.png',rgb,format='png')

    print("%i / %i faced detected" % (len(f) - fneg,len(f)))
    print(total)
    print(len(f))
    print("avg time: %.8f" % (total / len(f)))


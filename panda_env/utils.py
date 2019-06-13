
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from imutils import face_utils

class ShadowDetector():
    def __init__(self): return

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

class LM_Detector():
    def __init__(self):
        self.fa = cv2.dnn.readNetFromCaffe("../models/deploy.prototxt.txt","../models/res10_300x300_ssd_iter_140000.caffemodel")
        self.predictor5 = dlib.shape_predictor("../models/shape_predictor5.dat")
        self.thresh = 0.4

    #DLIB SIMPLE LANDMARK DETECTION + CPU YOLO FACE DETECTION
    def get_lm(self,rgb,pad=20):
        h,w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb,(300,300)),1.0,(300,300),(103.93,116.77,123.68))
        self.fa.setInput(blob)
        detections = self.fa.forward()

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

    #GET THE EYE REGION
    def get_eyes(self,rgb,lm,pad=40):
        vec1 = lm[0] - lm[2]
        x = -vec1[1] / vec1[0]
        y = vec1[0] * x / -vec1[1]
        vec2 = np.array([x,y])
        angle = -math.atan(vec2[1] / vec2[0])
        w = np.linalg.norm(vec1)
        h = 50
        cx = (lm[0][0] + lm[2][0]) // 2
        cy = (lm[0][1] + lm[2][1]) // 2
        rot_rect = ((cx,cy),(w+pad,h),angle)     #PADDED HEIGHT AND WIDTH OF 15PIXELS
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)

        eye_mask = np.zeros(rgb.shape[:2])
        cv2.fillConvexPoly(eye_mask,box,(1))

        return eye_mask == 1

    #FOR VISUALIZATION
    def view_lm(self,rgb,lm):
        for x,y in lm:
            rgb = cv2.circle(rgb,(x,y),2,[255,0,0],2)

        vec1 = lm[0] - lm[2]
        #ax + by = 0 let a,b = (1,1)
        x = -vec1[1] / vec1[0]
        y = vec1[0] * x / -vec1[1]
        vec2 = np.array([x,y])
        angle = -math.atan(vec2[1] / vec2[0])
        w = np.linalg.norm(vec1)
        h = 30
        cx = (lm[0][0] + lm[2][0]) // 2
        cy = (lm[0][1] + lm[2][1]) // 2
        rot_rect = ((cx,cy),(w+30,h),angle)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)

        cv2.fillConvexPoly(rgb,box,(255,255,255))
        cv2.drawContours(rgb,[box],0,(0,0,255),2)
        cv2.imshow('rgb',rgb)
        cv2.waitKey(0)
        quit()

    def draw(self,rgb,shadow,lm):
        IOU = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(np.logical_or(eye_mask,shadow))
        EYE = np.sum(np.logical_and(eye_mask,shadow)) / np.sum(eye_mask)



#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import numpy as np
import os.path
import sys
import json
import counter
import utils
import detection
import mask

from track import TrackerFacade
from ctypes import *


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.json', 'r') as handle:
    YOLOdict = json.load(handle)

mask = mask.Mask()
mask.loadMask("config-files/maskConfig.json")

counter = counter.loadCounter("config-files/counterConfig.json")


#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('videos/testRotonda.mp4')
    #Should we resize the video frame?
    #cap.set(3, 1280)
    #cap.set(4, 720)

    #Instance of tracker
    trackerFacade = TrackerFacade(50, 10)

    #Instance of simpleCounter
    counter.initCounter()

    #Saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

    net = load_net(b"darknet/cfg/yolov3.cfg", b"darknet/yolov3.weights", 0)
    meta = load_meta(b"darknet/cfg/coco.data")
    cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)

    roibbox = mask.getVertices()

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()

        detections = detection.detect_numpy(net, meta, img)                             #YOLO detection
        cleanedDetections = detection.cleanDetections(detections, roibbox, 0.8)         #Clean detections

        if(len(cleanedDetections) > 0):
            trackerFacade.update(cleanedDetections)                                     #Track detections
            centersVectors = trackerFacade.getCenterVector()                            #Get array of the last two centers for each object
            counter.count(centersVector)
            trackerFacade.draw(img, [0, 255, 0])

        mask.drawMask(img, [0,0,255])

        #counter.drawCounter(img)

        cv2.imshow("img", img)
        out.write(img)

        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

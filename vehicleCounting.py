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
import pickle
import counter
import utils

from tracker import Tracker
from ctypes import *
from detection import *


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)

mask = utils.Mask()
mask.loadMask("config-files/maskConfig.pickle")

counter = counter.loadCounter("config-files/counterConfig.pickle")


#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('videos/testSalou.mp4')
    #Should we resize the video frame?
    #cap.set(3, 1280)
    #cap.set(4, 720)

    #Instance of sort
    tracker = Tracker(160, 30, 5)

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

    roibbox = np.array(mask.getVertices())

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()
        roi = img[roibbox[0][1]:roibbox[1][1], roibbox[0][0]:roibbox[1][0]]

        r = detect_numpy(net, meta, roi)                          #YOLO detection

        if(len(r) > 0):
            tracker.update(r)                                     #Update tracking postion
            tracker.draw(roi, [0, 255, 0])                        #Draw tracked objects
            centers = tracker.getCenters()                        #Get centers of tracked objects
            counter.count(centers)                                #Count objects
            counter.drawLanes(roi)                                #Draw virtual lanes

        img[roibbox[0][1]:roibbox[1][1], roibbox[0][0]:roibbox[1][0]] = roi

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

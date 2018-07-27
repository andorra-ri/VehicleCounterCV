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

from ctypes import *
from detection import *

sys.path.insert(0, "sort/")
import sort


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
    mot_tracker = sort.Sort()

    #Instance of simpleCounter
    counter.initCounter()

    #Saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

    ret, img = cap.read()
    net = load_net(b"darknet/cfg/yolov3.cfg", b"darknet/yolov3.weights", 0)
    meta = load_meta(b"darknet/cfg/coco.data")
    cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)

    roibbox = np.array(mask.getVertices())

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()
        roi = img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]]

        r = detect_numpy(net, meta, roi)                          #YOLO detection
        track_bbs_ids = mot_tracker.update(r)                     #SORT tracking
        centers = mot_tracker.get_centers()                       #SORT object centers
        counter.count(centers)                                    #Counter count

        counter.drawLanes(roi)

        mot_tracker.draw(track_bbs_ids, roi, [0,255,0])
        img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]] = roi

        mask.drawMask(img, [0,0,255])

        counter.drawCounter(img)

        cv2.imshow("img", img)
        out.write(img)

        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

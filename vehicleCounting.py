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

from ctypes import *
from detection import *

sys.path.insert(0, "sort/")
import sort


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)

try:
    with open('config-files/maskGeom.pickle') as handle:
        mask = pickle.load(handle)
except:
    print("Oops! I couldn't find mask.pickle file... Make sure to run config.py to define it")

try:
    with open('config-files/counterGeom.pickle', 'rb') as handle:
        lanes = pickle.load(handle)
except:
    print("Oops! I couldn't find counter.pickle file... Make sure to run config.py to define it")

#-----------------------------
#<-------- Functions -------->
#-----------------------------
def drawRoi(bbox, img, color):
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])

    cv2.rectangle(img, pt1, pt2, color, 3)


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
    smplCounter = counter.simpleCounter()
    for lane in lanes:
        smplCounter.appendLane(lane)

    #Saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

    ret, img = cap.read()
    net = load_net(b"darknet/cfg/yolov3.cfg", b"darknet/yolov3.weights", 0)
    meta = load_meta(b"darknet/cfg/coco.data")
    cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)

    roibbox = np.array([mask[0][0], mask[0][1], mask[1][0], mask[1][1]])

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()
        roi = img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]]

        r = detect_numpy(net, meta, roi)                          #YOLO detection
        track_bbs_ids = mot_tracker.update(r)                     #SORT tracking
        centers = mot_tracker.get_centers()

        for center in centers:
            smplCounter.intersection(center)

        for lane in lanes:
            cv2.line(roi, tuple(lane[3][0][0]), tuple(lane[3][0][1]), [0,255,0], 2)

        mot_tracker.draw(track_bbs_ids, roi, [0,255,0])
        img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]] = roi

        drawRoi(roibbox, img, [0,0,255])
        
        cv2.imshow("img", img)
        out.write(img)

        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

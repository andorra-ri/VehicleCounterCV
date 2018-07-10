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

from ctypes import *
from sort import *
from detection import *

sys.path.insert(0, "sort/")
import sort


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)


#-----------------------------
#<-------- Functions -------->
#-----------------------------
def convertBack(x, y, w, h):
	xmin = int(round(x - (w/2)))
	xmax = int(round(x + (w/2)))
	ymin = int(round(y - (h/2)))
	ymax = int(round(y + (h/2)))
	return xmin, ymin, xmax, ymax

def draw(bboxs, img, color):
    for i in bboxs:
            xmin, ymin, xmax, ymax = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            if len(i) == 6:
                otype = list(YOLOdict.keys())[list(YOLOdict.values()).index(i[5])]
                cv2.rectangle(img, pt1, pt2, color, 2)
                cv2.putText(img, otype+str(i[4]), (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

def drawRoi(bbox, img, color):
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])

    cv2.rectangle(img, pt1, pt2, color, 3)


#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('testSalouS.mov')
    #Should we resize the video frame?
    #cap.set(3, 1280)
    #cap.set(4, 720)

    #Instance of sort
    mot_tracker = sort.Sort()

    #Saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

    ret, img = cap.read()
    net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    #Define roi as [xmin, ymin, xmax, ymax]
    roibbox = np.array([350,240,750, 390])

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()
        roi =img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]]
        r = detect_numpy(net, meta, roi)                          #YOLO detection
        track_bbs_ids = mot_tracker.update(r)                     #SORT tracking
        draw(track_bbs_ids, roi, [0,255,0])
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

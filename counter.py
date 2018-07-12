#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import numpy as np

#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)


#-----------------------------
#<--------- Classes --------->
#-----------------------------
class lane:
    def __init__(self, id, name, type):
        self.ID = id
        self.NAME = name
        self.VERTICES = []              #[[x1, y1], [x2, y2]]
        self.TYPE = type                      #type must be 0(indiferent), 1(in), 2(out)

    def appendVertices(self, vertices):
        self.VERTICES.append(vertices)

    def getVertices(self):
        return self.VERTICES



class simpleCounter:
    def __init__(self, id, name):
        self.ID = id
        self.NAME = name
        self.lanes = []
        self.counter = {}            #Counter with the following structure: {type:counts, type:counts, ...}

        for key, value in YOLOdict.items():
            self.counter[value] = 0

    def appendLane(self, lane):
        self.lanes.append(lane)

    def intersection(self, centers):

        line1 = line(centers[2], centers[3])

        for lane in lanes:
            line2 = line(lane[0], lane[1])

            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                count(centers[1])

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def count(self, type):
        self.counter[type] +=1

    def getCounts(self):
        return self.counter

    def drawLanes(self, img):
        if(len(lanes) > 0):
            for lane in lanes:
                cv2.line(img, tuple(lane[3][0][0]), tuple(lane[3][0][1]), [0,255,0], 1)

    def clear(self):
        self.counter = dict.fromkeys(self.counter, 0)



class complexCounter:
    #empty for the moment

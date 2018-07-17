#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import pickle

from matplotlib.path import Path
from numpy import dot, array, empty_like

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
        self.counter = []
        self.dictCounter = {}            #Counter with the following structure: {type:counts, type:counts, ...}

        for key, value in YOLOdict.items():
            self.counter[value] = 0

    def appendLane(self, lane):
        self.lanes.append(lane)

    def initCounter(self):
        self.counter = [self.dictCounter.copy() for k in range(len(self.lanes))]

    @staticmethod
    def make_path(x1, y1, x2, y2):
        return Path([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])

    @staticmethod
    def perp(a):
        b = empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    def intersection(self, centers):
        a1 = array(centers[2])
        a2 = array(centers[3])

        for l, lane in enumerate(self.lanes):
            b1 = array(lane[3][0][0])
            b2 = array(lane[3][0][1])

            da = a2-a1
            db = b2-b1
            dp = a1-b1
            dap = self.perp(da)
            denom = dot( dap, db)
            num = dot( dap, dp )

            x3 = ((num / denom.astype(float))*db + b1)[0]
            y3 = ((num / denom.astype(float))*db + b1)[1]
            p1 = self.make_path(a1[0],a1[1],a2[0],a2[1])
            p2 = self.make_path(b1[0],b1[1],b2[0],b2[1])
            if p1.contains_point([x3,y3]) and p2.contains_point([x3,y3]):
                self.count(l, centers[1])

    def count(self, l, type):
        self.counter[l][type] +=1

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

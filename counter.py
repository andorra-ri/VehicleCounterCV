#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import json
import os.path


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.json', 'r') as handle:
    YOLOdict = json.load(handle)


#-----------------------------
#<--------- Classes --------->
#-----------------------------
class Lane:
    def __init__(self, id, name, type, vertices):
        self.ID = id
        self.NAME = name
        self.VERTICES = vertices              #[[x1, y1], [x2, y2]]
        self.TYPE = type                      #type must be 0(indiferent), 1(in), 2(out)


    def setVertices(self, vertices):
        self.VERTICES = vertices


    def getVertices(self):
        return self.VERTICES



class SimpleCounter:
    def __init__(self, id, name):
        self.ID = id
        self.NAME = name
        self.lanes = []
        self.lanesCounter = []
        self.dictCounter = {}            #Counter with the following structure: {type:counts, type:counts, ...}

        for key, value in YOLOdict.items():
            self.dictCounter[value] = 0


    def appendLane(self, lane):
        self.lanes.append(lane)


    def initCounter(self):
        self.lanesCounter = [self.dictCounter.copy() for k in range(len(self.lanes))]


    def addCount(self, lane, type):
        for l, ln in enumerate(self.lanes):
            if (ln == lane):
                self.lanesCounter[l][type] +=1


    #MAIN method of the class
    def count(self, centers):
        for vector in centers:
            lane = utils.laneIntersection(vector, self.lanes)
            self.addCount(lane, vector[1])


    def getCounts(self):
        return self.lanesCounter


    def drawLanes(self, img):
        if (len(self.lanes) > 0):
            for lane in self.lanes:
                cv2.line(img, tuple(lane[3][0][0]), tuple(lane[3][0][1]), [0,255,0], 1)


    def drawCounter(self, img):
        for t, count in enumerate(self.lanesCounter):
            yAddjust = t * 350

            cv2.rectangle(img, (img.shape[1]-250, 50+yAddjust), (img.shape[1]-50, 50+300+yAddjust), color, -1)
            cv2.putText(img, "Counter", (img.shape[1]-230, 100+yAddjust), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 6)

            for num, key in enumerate(count):
                keyStr = str(list(YOLOdict.keys())[list(YOLOdict.values()).index(key)])
                value = str(self.lanesCounter.get(key))
                cv2.putText(img, keyStr+": "+value, (img.shape[1]-230, 160+40*num+yAddjust), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 4)


    def clear(self):
        self.lanesCounter = dict.fromkeys(self.lanesCounter, 0)


    def storeToMySQL(self):




class TrackingCounter:
    def __init__(self, id, name):
        self.ID = id
        self.NAME = name
        self.lanes_in = []
        self.lanes_out = []
        self.combinationLanes = []
        self.counter = []
        self.dictCounter = {}            #Counter with the following structure: {type:counts, type:counts, ...}
        self.idsInside = []

        for key, value in YOLOdict.items():
            self.dictCounter[value] = 0


    def appendLane(self, lane):
        if (lane.TYPE == 1):
            self.lanes_in.append(lane)
        elif (lane.TYPE == 2):
            self.lanes_out.append(lane)


    def initCounter(self):
        for lane_in in self.lanes_in:
            for lane_out in self.lanes_out:
                self.combinationLanes.append([lane_in, lane_out])

        self.counter = [self.dictCounter.copy() for k in range(len(self.combinationLanes))]


    def manage(self, center):
        if (center[0] in self.idsInside):
            lane_out = utils.laneIntersection(center, self.lanes_out)
            if (not lane_out):
                for id in self.idsInside:
                    if (id == center[0]):
                        lane_in = id[1]
                        type = center[1]
                        self.addCount(lane_in, lane_out, type)    #we should pass lane_in, lane_out and type
                self.idsInside.pop(center[0])

        else:
            lane_in = utils.laneIntersection(center, self.lanes_in)
            if (not lane_in):
                ids.IdsInside.append([center[0], lane_in])


    def addCount(self, lane_in, lane_out, type):
        for l, combination in self.combinationLanes:
            if (combination[0] == lane_in and combination[1] == lane_out):
                self.counter[l][type] += 1


    #MAIN method of the class
    def count(self, centers):
        for center in centers:
            self.manage(center)


    def getCounts(self):
        return self.counter


    def drawLanes(self, img):
        if (len(self.lanes_in) > 0):
            for laneIn in self.lanes_in:
                cv2.line(img, tuple(laneIn.VERTICES[0]), tuple(laneIn.VERTICES[1]), [0,255,0], 1)
        if (len(self.lanes_out) > 0):
            for laneOut in self.lanes_out:
                cv2.line(img, tuple(laneOut.VERTICES[0]), tuple(laneOut.VERTICES[1]), [0,0,255], 1)



def loadCounter(path):
    if (os.path.exists(path)):
        with open(path, 'r') as handle:
            counterConfig = json.load(handle)
            if (counterConfig[2] == 0):
                counter = SimpleCounter(counterConfig[0], counterConfig[1])
            else:
                counter = ComplexCounter(counterConfig[0], counterConfig[1])

            for lane in counterConfig[3]:
                counter.appendLane( Lane(lane[0], lane[1], lane[2], lane[3]) )

    else:
        counter = []

    return counter

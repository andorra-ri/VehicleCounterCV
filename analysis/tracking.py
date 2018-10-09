#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import sys
import cv2
import json
import analyzer
import geometries
import importlib

sys.path.insert(1, "../")
import sqlmanager
import utils

#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('../config-files/YOLOobjects.json', 'r') as handle:
    fileConfig = json.load(handle)
    YOLOobjects = fileConfig["YOLOobjects"]


class Tracking:
    def __init__(self, jsonConfig):
        self.ID = jsonConfig["id"]
        self.NAME = jsonConfig["name"]
        self.geomType = jsonConfig["geomType"]
        self.geom_in = []
        self.geom_out = []
        self.idsInside = []
        self.dictCounter = dict.fromkeys(YOLOobjects, 0)

        self.appendGeometry(jsonConfig["geomConfig"])
        self.initCounter()


    def appendGeometry(self, rawGeometry):
        module = importlib.import_module("geometries")
        class_ = getattr(module, self.geomType.title())

        for geometry in rawGeometry:
            geomObjt = class_(rawGeometry["id"], rawGeometry["name"], rawGeometry["type"], rawGeometry["vertices"])
            if(geomObjt.TYPE == 1):
                self.geom_in.append(geomObjt)
            elif(geomObjt.TYPE == 2):
                self.geom_out.append(geomObjt)


    def initCounter(self):
        for gm_in in self.geom_in:
            for gm_out in self.geom_out:
                self.combinationGeometries.append([gm_in, gm_out])

        self.counter = [self.dictCounter.copy() for k in range(len(self.combinationGeometries))]


    def manage(self, center):
        if (center[0] in self.idsInside):
            geom_out = utils.laneIntersection(center, self.geom_out)
            if (not geom_out):
                for id in self.idsInside:
                    if (id == center[0]):
                        geom_in = id[1]
                        type = center[1]
                        self.addCount(geom_in, geom_out, type)
                self.idsInside.pop(center[0])

        else:
            geom_in = utils.laneIntersection(center, self.geom_in)
            if (not geom_in):
                ids.IdsInside.append([center[0], geom_in])


    def addCount(self, geom_in, geom_out, type):
        for l, combination in self.combinationGeometries:
            if (combination[0] == geom_in and combination[1] == geom_out):
                self.counter[l][type] += 1


    #MAIN method of the class
    def main(self, classTrackers):
        centers = classTrackers.getCentersVector(2)
        for center in centers:
            self.manage(center)


    def getCounts(self):
        return self.counter


    def drawGeometries(self, img, color):
        for geomIn in self.geom_in:
            geomIn.draw(img, color)
        for geomOut in self.geom_out:
            geomOut.draw(img, color)


    def saveToSQL(self):
        #ToDo

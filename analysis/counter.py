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
with open('config-files/YOLOobjects.json', 'r') as handle:
    fileConfig = json.load(handle)
    YOLOobjects = fileConfig["YOLOobjects"]


class Counter(Analyzer):
    def __init__(self, jsonConfig):
        self.ID = jsonConfig["id"]
        self.NAME = jsonConfig["name"]
        self.geomType = jsonConfig["geomType"]
        self.counter = []
        self.geometries = []
        self.dictCounter = dict.fromkeys(YOLOobjects, 0) #Counter with the following structure: {type:counts, type:counts, ...}

        self.appendGeometry(jsonConfig["geomConfig"])
        self.initCounter()


    def appendGeometry(self, rawGeometry):
        module = importlib.import_module("geometries")
        clss_ = getattr(module, self.geomType.title())

        for geometry in rawGeometry:
            self.geometries.append( clss_(geometry["id"], geometry["name"], geometry["type"], geometry["vertices"]) )


    def initCounter(self):
        self.counter = []
        for k in range(len(self.geometries)):
            geomDict = self.dictCounter.copy()
            geomDict["id"] = self.geometries[k].ID
            self.counter.append(geomDict)


    def addCount(self, geomID, type):
        for g, gm in enumerate(self.geometries):
            if (gm.ID == geomID):
                self.counter[g][type] +=1


    #MAIN method of the class
    def main(self, classTrackers):
        if(self.geomType == "lane"):
            centers = classTrackers.getCentersVector(2)
            for vector in centers:
                geom = utils.laneIntersection(vector[2], self.geometries)
                self.addCount(geom, vector[1])
        elif(self.geomType == "zone"):
            centers = classTrackers.getCentersVector(1)
            for vector in centers:
                geom = utils.zoneContains(vector[2], self.geometries)
                self.addCount(geom, vector[1])

    def getCounts(self):
        return self.counter


    def drawGeometries(self, img, color):
        for geom in self.geometries:
            geom.draw(img, color)


    def draw(self, img):
        for t, count in enumerate(self.counter):
            yAddjust = t * 350

            cv2.rectangle(img, (img.shape[1]-250, 50+yAddjust), (img.shape[1]-50, 50+300+yAddjust), [100, 100, 100], -1)
            cv2.putText(img, "Counter", (img.shape[1]-230, 100+yAddjust), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 6)

            for num, key in enumerate(count):
                value = str(count[key])
                cv2.putText(img, key+": "+value, (img.shape[1]-230, 160+40*num+yAddjust), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 4)


    def clear(self):
        self.counter = dict.fromkeys(self.counter, 0)


    def saveToSQL(self, instanceSQLManager):
        table = instanceSQLManager.getTable()
        dictKeys = list(self.dictCounter.keys())
        keysComSep = ', '.join(dictKeys)
        keysVal = ')s , %('.join(dictKeys)

        sqlStatement = "INSERT INTO " + table + " (id, timestamp, " + keysComSep + ") VALUES ( %(id)s, NOW(), %(" + keysVal + ")s )"

        instanceSQLManager.executeInsertQuery(sqlStatement, self.counter)

        self.clear()

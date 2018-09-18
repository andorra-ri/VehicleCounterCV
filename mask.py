#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import numpy as np
import json
import os.path
import cv2
import math


#-----------------------------
#<--------- Classes --------->
#-----------------------------
class Mask:
    def __init__(self):
        self.VERTICES = []              #[[xmin,ymin], [xmax, ymax]]


    def loadMask(self, path):
        if(os.path.exists(path)):
            with open(path, 'r') as handle:
                maskConfig = json.load(handle)
                self.setVertices(maskConfig)

        return self.VERTICES


    def saveMask(self, path):
        with open(path, 'w') as handle:
            json.dump(self.VERTICES, handle)


    def getVertices(self):
        return self.VERTICES


    def setVertices(self, vertices):
        self.VERTICES = vertices


    def drawMask(self, img, color):
        if(len(self.VERTICES) > 0):
            pt1 = (self.VERTICES[0][0], self.VERTICES[0][1])
            pt2 = (self.VERTICES[1][0], self.VERTICES[1][1])

            cv2.rectangle(img, pt1, pt2, color, 3)

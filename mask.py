#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import numpy as np
import pickle
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
            with open(path, 'rb') as handle:
                maskConfig = pickle.load(handle)
                self.setVertices(maskConfig)

        return self.VERTICES


    def saveMask(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.VERTICES, handle, protocol = pickle.HIGHEST_PROTOCOL )


    def getVertices(self):
        return self.VERTICES


    def setVertices(self, vertices):
        self.VERTICES = vertices


    def drawMask(self, img, color):
        if(len(self.VERTICES) > 0):
            pt1 = (self.VERTICES[0][0], self.VERTICES[0][1])
            pt2 = (self.VERTICES[1][0], self.VERTICES[1][1])

            cv2.rectangle(img, pt1, pt2, color, 3)

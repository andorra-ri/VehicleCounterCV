#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import numpy as np
import pickle
import os.path
import cv2
import math

#-----------------------------
#<-------- Functions -------->
#-----------------------------
def bboxToCenter(bbox):       #bbox = [xmin, ymin, xmax, ymax]
    x, y = bbox[0], bbox[1]
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    center = [x + width/2, y + height/2]

    return center

def distanceBetweenTwoPoints(point1, point2):
    diff = [point1[0]-point2[0], point1[1]-point2[1]]
    distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

    return distance

def moduleVector(vector):
    module = np.sqrt(vector[0]**2 + vector[1]**2)

    return module

def cosinusBetweenTwoVectors(referenceVector, testVector):

    dot = referenceVector[0]*testVector[0] + referenceVector[1]*testVector[1]
    det = (moduleVector(referenceVector) * moduleVector(testVector))

    cos = dot/det

    return(cos)



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
                self.appendVertices(maskConfig)

        return self.VERTICES

    def saveMask(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.VERTICES, handle, protocol = pickle.HIGHEST_PROTOCOL )

    def getVertices(self):
        return self.VERTICES

    def appendVertices(self, vertices):
        self.VERTICES = vertices

    def drawMask(self, img, color):
        if(len(self.VERTICES) > 0):
            pt1 = (self.VERTICES[0][0], self.VERTICES[0][1])
            pt2 = (self.VERTICES[1][0], self.VERTICES[1][1])

            cv2.rectangle(img, pt1, pt2, color, 3)

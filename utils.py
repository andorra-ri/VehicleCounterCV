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
    

def insideBbox(bboxReference, bboxTest):
    if(bboxTest[0] < bboxReference[0] or bboxTest[1] < bboxReference[1] or bboxTest[2] > bboxReference[2] or bboxTest[3] > bboxReference[3]):
        return False
    else:
        return True


def distanceBetweenTwoPoints(point1, point2):
    diff = [point1[0]-point2[0], point1[1]-point2[1]]
    distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

    return distance


def moduleVector(vector):
    module = np.sqrt(vector[0]**2 + vector[1]**2)

    return module


def cosineBetweenTwoVectors(referenceVector, testVector):

    dot = referenceVector[0]*testVector[0] + referenceVector[1]*testVector[1]
    det = (moduleVector(referenceVector) * moduleVector(testVector))

    cos = dot/det

    return(cos)

def iou(bb1,bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(bb1[0], bb2[0])
	yA = max(bb1[1], bb2[1])
	xB = min(bb1[2], bb2[2])
	yB = min(bb1[3], bb2[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
	boxBArea = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

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

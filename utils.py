#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import numpy as np
import math
from matplotlib.path import Path

#-----------------------------
#<-------- Functions -------->
#-----------------------------
def bboxToCenter(bbox):       #bbox = [xmin, ymin, xmax, ymax]
    x, y = bbox[0], bbox[1]
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    center = [x + width/2, y + height/2]

    return center


def insideBbox(bboxReference, bboxTest):
    if(bboxTest[0] < bboxReference[0][0] or bboxTest[1] < bboxReference[0][1] or bboxTest[2] > bboxReference[1][0] or bboxTest[3] > bboxReference[1][1]):
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


def make_path(x1, y1, x2, y2):
    return Path([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def laneIntersection(self, vector, lanes):
    a1 = np.array(vector[0])
    a2 = np.array(vector[1])

    for lane in lanes:
        laneVertices = lane.getVertices()
        b1 = np.array(laneVertices[0])
        b2 = np.array(laneVertices[1])

        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = np.dot( dap, db)
        num = np.dot( dap, dp )

        x3 = ((num / denom.astype(float))*db + b1)[0]
        y3 = ((num / denom.astype(float))*db + b1)[1]
        p1 = make_path(a1[0],a1[1],a2[0],a2[1])
        p2 = make_path(b1[0],b1[1],b2[0],b2[1])
        if (p1.contains_point([x3,y3]) and p2.contains_point([x3,y3])):
            return lane


def zoneContainsPoint(self, point, zones):
    pointX = point[0]
    pointY = point[1]

    for zone in zones:
        zoneVertices = zone.getVertices()
        zoneLeftTop = zoneVertices[0]
        zoneRightBottom = zoneVertices[1]

        if(pointX >= zoneLeftTop[0] and pointX <= zoneRightBottom[0] and pointY >= zoneLeftTop[1] and pointY <= zoneRightBottom[1]):
            return zone

#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import cv2
import numpy as np
import utils
import pickle
from scipy.optimize import linear_sum_assignment


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)


#-----------------------------
#<--------- Classes --------->
#-----------------------------
class Tracker:

        counter = 0	 # Total vehicles count

        def __init__(self, bbox, type):
            Tracker.counter += 1
            self.ID = Tracker.counter
            self.TYPE = type
            self.centers = []
            self.bbox = []
            self.predictions = []
            self.skippedFrames = 0

            self.bbox.append(bbox)
            center = utils.bboxToCenter(bbox)

            # Init Kalman Filter for tracking
            self.kalman = cv2.KalmanFilter(4,2)
            self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
            self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
            self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
            self.kalman.statePre = np.array([[center[0]],[center[1]],[0],[0]],np.float32)

            self.update(center)


        # Update vehicle's properties
        def update(self, center):
            self.centers.append(center)
            if len(self.centers) > 10:
                del self.centers[0]
            self.predictNext()


        # Calculate distance from vehicle's center to a point
        def dist(self, point):
            return math.hypot(point[0] - self.centers[-1][0], point[1] - self.centers[-1][1])


        # Return the last center available
        def center(self):
            return self.centers[-1]


        # Return the last center prediction available
        def prediction(self):
            return self.predictions[-1]


        # Predict next vehicle's position using Kalman Filter
        def predictNext(self):
            center = np.array([[np.float32( self.centers[-1][0] )],[np.float32( self.centers[-1][1] )]])
            self.kalman.correct(center)
            tp = self.kalman.predict()
            self.predictions.append((int(tp[0]),int(tp[1])))
            if len(self.predictions) > 10:
                del self.predictions[0]


        # Draw vehicle's properties
        def draw(self, img, color):
            xmin, ymin, xmax, ymax = int(self.bbox[-1][0]), int(self.bbox[-1][1]), int(self.bbox[-1][2]), int(self.bbox[-1][3])
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, color, 1)
            cv2.circle(img, (int(self.centers[-1][0]), int(self.centers[-1][1])), 2, (0,0,255))
            otype = list(YOLOdict.keys())[list(YOLOdict.values()).index(self.TYPE)]
            cv2.putText(img, str(otype)+'id: '+str(self.ID), (int(self.centers[-1][0]), int(self.centers[-1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
            '''for i in range(1, len(self.centers)):
                cv2.line(img, (int(self.centers[i-1][0]), int(self.centers[i-1][1])), (int(self.centers[i][0]), int(self.centers[i][1])), (0,0,255), 1)
            for i in range(1, len(self.predictions)):
                cv2.line(img, (int(self.predictions[i-1][0]), int(self.predictions[i-1][1])), (int(self.predictions[i][0]), int(self.predictions[i][1])), (0,255,255), 1)
                #if len(self.predictions) > 0: cv2.circle(frame, (int(self.predictions[-1][0]), int(self.predictions[-1][1])), maxDist, (0,255,255))'''


class TrackerFacade:

        def __init__(self, maxDist, maxFramesSkipped):
            self.maxDist = maxDist
            self.maxFramesSkipped = maxFramesSkipped
            self.trackers = []


        def iouCostMatrix(self, detections):
            N = len(self.trackers)
            M = len(detections)
            cost = np.zeros(shape=(N, M))
            for i in range(len(self.trackers)):
                for i in range(len(detections)):
                    xx1 = np.maximum(detections[j][0], self.trackers[i].bbox[-1][0])
                    yy1 = np.maximum(detections[j][1], self.trackers[i].bbox[-1][1])
                    xx2 = np.minimum(detections[j][2], self.trackers[i].bbox[-1][2])
                    yy2 = np.minimum(detections[j][3], self.trackers[i].bbox[-1][3])
                    w = np.maximum(0., xx2 - xx1)
                    h = np.maximum(0., yy2 - yy1)
                    wh = w * h
                    o = wh / ((detections[j][2]-detections[j][0])*(detections[j][3]-detections[j][1])
                      + (self.trackers[i].bbox[-1][2]-self.trackers[i].bbox[-1][0])*(self.trackers[i].bbox[-1][3]-self.trackers[i].bbox[-1][1]) - wh)

                    cost[i][j] = o

            return cost


        def distanceCosineCostMatrix(self, detections):
            N = len(self.trackers)
            M = len(detections)
            cost = np.zeros(shape=(N, M))
            for i in range(len(self.trackers)):
                for j in range(len(detections)):

                        detectionCenter = utils.bboxToCenter(detections[j][:4])
                        predictionCenter = self.trackers[i].predictions[-1]
                        previousCenter = self.trackers[i].center()

                        referenceVector = predictionCenter - previousCenter
                        testVector = detectionCenter - previousCenter

                        distance = utils.distanceBetweenTwoPoints(centerDetection, centerPrediction)
                        cos = utils.cosineBetweenTwoVectors(referenceVector, testVector)

                        if(cos > 0):
                            value = distance * (2-cos)
                        else:
                            value = 100000

                        cost[i][j] = distance

            return cost


        def update(self, detections):

            if (len(self.trackers) == 0):
                for detection in detections:
                    vehicle = Tracker(detection[:4], detection[4])
                    self.trackers.append(vehicle)

            else:

                costMatrix = self.distanceCosinusCostMatrix(detections)
                row_ind, col_ind = linear_sum_assignment(costMatrix)      # Hungarian method for assignment


                assignment = np.full(len(self.trackers), -1, dtype = int)
                for i in range(len(row_ind)):
                    assignment[row_ind[i]] = col_ind[i]


                # Identify trackers with no assignment, if any
                un_assigned_trackers = []
                for i in range(len(assignment)):
                    if (assignment[i] != -1):
                        # If cost is very high then un_assign (delete) the track
                        if (costMatrix[i][assignment[i]] > self.maxDist):
                            assignment[i] = -1
                            un_assigned_trackers.append(i)
                        pass
                    else:
                        self.trackers[i].skippedFrames += 1


                # If trackers are not detected for long time, remove them
                del_trackers = []
                for i in range(len(self.trackers)):
                    if (self.trackers[i].skippedFrames > self.maxFramesSkipped):
                        del_trackers.append(i)
                if len(del_trackers) > 0:  # only when skipped frame exceeds max
                    for id in del_trackers:
                        del self.trackers[id]
                        del assignment[id]


                # Now look for un_assigned detects
                un_assigned_detects = []
                for i in range(len(detections)):
                        if i not in assignment:
                            un_assigned_detects.append(i)


                # Start new trackers
                if(len(un_assigned_detects) != 0):
                    for i in range(len(un_assigned_detects)):
                        vehicle = Tracker(detections[un_assigned_detects[i]][:4], detections[un_assigned_detects[i]][4])
                        self.trackers.append(vehicle)


                # Update KalmanFilter state, lastResults and trackers trace
                for i in range(len(assignment)):
                    if(assignment[i] != -1):
                        self.trackers[i].skippedFrames = 0
                        bbox = detections[assignment[i]][:4]
                        center = utils.bboxToCenter(bbox)

                        self.trackers[i].bbox.append(bbox)
                        self.trackers[i].update(center)
                    else:
                        center = self.trackers[i].predictions[-1]
                        self.trackers[i].update(center)


        def getCentersVector():
            centersVector = []
            for trk in self.trackers:
                centersVector.append([trk.id, trk.type, trk.centers[-1], trk.centers[-2]])

            return centersVector


        def draw(self, img, color):
            for vehicle in self.trackers:
                vehicle.draw(img, color)

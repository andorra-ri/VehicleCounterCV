'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan - github.com/srianant/kalman_filter_multi_object_tracking
    Modified          : Guillem Francisco
    Date created      : 07/14/2017
    Date last modified: 08/09/2018
    Python Version    : 3.5
'''

#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import numpy as np
import utils
import cv2
import pickle
from kalman_filter import KalmanFilter
from common import dprint
from scipy.optimize import linear_sum_assignment


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)


#-----------------------------
#<--------- Classes --------->
#-----------------------------
class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """
    counter = 0

    def __init__(self, center, bbox, type):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
        Return:
            None
        """
        Track.counter += 1
        self.ID = Track.counter                       # identification of each track object
        self.TYPE = type
        self.KF = KalmanFilter()                      # KF instance to track this object
        self.prediction = np.asarray(center)          # predicted centroids (x,y)
        self.skipped_frames = 0                       # number of frames skipped undetected
        self.trace = []                               # trace path
        self.bbox = []
        self.bbox.append(bbox)


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []

    def update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected object with following structure [xmin, ymin, xmax, ymax, typeObject]
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                bbox = detections[i][:4]
                center = utils.bboxToCenter(bbox)
                type = detections[i][4]
                track = Track(center, bbox, type)
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    center = utils.bboxToCenter(detections[j][:4])
                    diff = self.tracks[i].prediction - center
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                bbox = detections[un_assigned_detects[i]][:4]
                center = utils.bboxToCenter(bbox)
                type = detections[un_assigned_detects[i]][4]
                track = Track(center, bbox, type)
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                bbox = detections[assignment[i]][:4]
                center = utils.bboxToCenter(bbox)
                self.tracks[i].prediction = self.tracks[i].KF.correct(center, 1)
                self.tracks[i].bbox.append(bbox)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

    def draw(self, img, color):
        """Draw bbox of tracked objects:
            - Draw bbox as a rectangle for each obejct
            - Draw ID and type of object as a text above the bbox
        Args:
            img: cv2 image in which to draw
            color: color to draw bbox
        Return:
            None
        """

        for i in range(len(self.tracks)):
            bbox = self.tracks[i].bbox[-1]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            otype = list(YOLOdict.keys())[list(YOLOdict.values()).index(self.tracks[i].TYPE)]
            cv2.rectangle(img, pt1, pt2, color, 1)
            cv2.putText(img, otype+str(self.tracks[i].ID), (pt1[0], pt1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 6)

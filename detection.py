#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import numpy as np
import json
import utils

from ctypes import *


#-----------------------------
#<------ Configuration ------>
#-----------------------------
with open('config-files/YOLOobjects.json', 'r') as handle:
    fileConfig = json.load(handle)
    YOLOobjects = fileConfig["YOLOobjects"]


#-----------------------------
#<-------- YOLO shit -------->
#-----------------------------
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    new_values = values.ctypes.data_as(POINTER(ctype))
    return new_values


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res

def detectObjects(net, meta, image, thresh=.3, hier_thresh=.5, nms=.45):
    im, arr = array_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if (dets[j].prob[i] > 0 & (meta.names[i].decode('utf-8') in YOLOobjects)):          #probably we will need to add decode('utf-8') to meta.names
                b = dets[j].bbox
                res.append([int(b.x - b.w/2), int(b.y - b.h/2), int(b.x + b.w/2), int(b.y + b.h/2), str(meta.names[i].decode('utf-8')), float(dets[j].prob[i])])
    free_detections(dets, num)
    return res


def detectionsInsideBbox(maskBbox, detections):
    insideDetections = []
    for det in detections:
        if(utils.insideBbox(maskBbox, det)):
            insideDetections.append(det)

    return insideDetections

def iouBetweenDetections(detections, iouThreshold):
    for i in reversed(range(len(detections))):
        for j in reversed(range(i)):
            iou = utils.iou(detections[i][:4], detections[j][:4])
            if(iou > iouThreshold):
                if(detections[i][5]>detections[j][5]):
                    detections.pop(j)
                else:
                    detections.pop(i)

    return detections


def cleanDetections(detections, maskBbox, iouThreshold):
    cleanedDetections = detectionsInsideBbox(maskBbox, detections)
    cleanedDetections = iouBetweenDetections(cleanedDetections, iouThreshold)

    return cleanedDetections


def drawDetections(bboxs, img, color):
    for bbox in bboxs:
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if(len(bbox) == 6):
            cv2.rectangle(img, pt1, pt2, color, 1)
            cv2.putText(img, str(bbox[4]), (pt1[0], pt1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 6)

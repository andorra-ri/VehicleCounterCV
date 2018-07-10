#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import numpy as np
import os.path
import sys
import pickle

from ctypes import *
from sort import *

sys.path.insert(0, "sort/")
import sort


#-----------------------------
#<-------- Functions -------->
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

with open('YOLOdict.pickle', 'rb') as handle:
    YOLOdict = pickle.load(handle)

#lib = CDLL("/home/nvidia/Desktop/darknet/darknetPython/libdarknet.so", RTLD_GLOBAL) #Absolute path
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
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

def detect_numpy(net, meta, image, thresh=.3, hier_thresh=.5, nms=.45):
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
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                intType = YOLOdict.get(meta.names[i].decode('utf-8'))
                if(intType == None): intType = 5
                res.append(((b.x - b.w/2), (b.y - b.h/2), (b.x + b.w/2), (b.y + b.h/2), dets[j].prob[i], intType))
                #res.append((int(b.x - b.w/2), int(b.y - b.h/2), int(b.x + b.w/2), int(b.y + b.h/2), str(meta.names[i])))
    free_detections(dets, num)
    r =np.array(res)
    return r

def convertBack(x, y, w, h):
	xmin = int(round(x - (w/2)))
	xmax = int(round(x + (w/2)))
	ymin = int(round(y - (h/2)))
	ymax = int(round(y + (h/2)))
	return xmin, ymin, xmax, ymax

def draw(bboxs, img, color):
    for i in bboxs:
            xmin, ymin, xmax, ymax = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            if len(i) == 6:
                otype = list(YOLOdict.keys())[list(YOLOdict.values()).index(i[5])]
                cv2.rectangle(img, pt1, pt2, color, 2)
                cv2.putText(img, otype+str(i[4]), (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

def drawRoi(bbox, img, color):
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])

    cv2.rectangle(img, pt1, pt2, color, 3)



#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('testSalouS.mov')
    #Should we resize the video frame?
    #cap.set(3, 1280)
    #cap.set(4, 720)

    #Instance of sort
    mot_tracker = sort.Sort()

    #Saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

    ret, img = cap.read()
    net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    #Define roi as [xmin, ymin, xmax, ymax]
    roibbox = np.array([350,240,750, 390])

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()
        roi =img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]]
        r = detect_numpy(net, meta, roi)                          #YOLO detection
        track_bbs_ids = mot_tracker.update(r)                     #SORT tracking
        draw(track_bbs_ids, roi, [0,255,0])
        img[roibbox[1]:roibbox[3], roibbox[0]:roibbox[2]] = roi
        drawRoi(roibbox, img, [0,0,255])
        cv2.imshow("img", img)
        out.write(img)

        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

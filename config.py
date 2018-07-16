#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import math
import random
import cv2
import time
import numpy as np
import pickle
import os.path
import counter


#-----------------------------
#<------ Configuration ------>
#-----------------------------
instructions = '[m] Add mask or [c] Add counter'
extra = ''
step = 0
counterTYPE = ''
counterID = ''
counterNAME = ''
laneID = ''
laneNAME = ''
laneTYPE = ''
laneVERTICES = []

if(os.path.exists("config-files/maskGeom.pickle")):
    with open('config-files/maskGeom.pickle', 'rb') as handle:
        maskVertices = pickle.load(handle)
else:
    maskVertices = []

if(os.path.exists("config-files/counterGeom.pickle")):
    with open('config-files/counterGeom.pickle', 'rb') as handle:
        lanes = pickle.load(handle)
else:
    lanes = []


#-----------------------------
#<-------- Functions -------->
#-----------------------------
# MOUSE EVENT HANDLER
def mouseClicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if step == 1:
            if(len(maskVertices) <= 2):
                maskVertices.append([x,y])
            elif(len(maskVertices) == 2):
                print("Mask has two vertices, you can't add another vertice!")
        if step == 9:
            if(len(laneVERTICES) <= 2):
                xmask = maskVertices[0][0]
                ymask = maskVertices[0][1]
                xlane = x - xmask
                ylane = y - ymask
                laneVERTICES.append([xlane,ylane])
            elif(len(laneVERTICES) ==2):
                print("Lane has two vertices, you can't add another vertice!")


#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('videos/testSalou.mp4')
    ret, img = cap.read()
    cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('img', mouseClicked)

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()

        if(len(maskVertices) == 2):
            cv2.rectangle(img, tuple(maskVertices[0]), tuple(maskVertices[1]), [0,0,255], 2)

        if(len(lanes) > 0):
            for lane in lanes:
                cv2.line(img, tuple(lane[3][0][0]), tuple(lane[3][0][1]), [0,255,0], 1)

        cv2.putText(img, instructions, (20,30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
	    cv2.putText(img, extra, (20,55), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))

        cv2.imshow("img", img)

        key = cv2.waitKey(1) & 0xFF
        #Idle
    	if step == 0:
    		extra = ''
    		if key == 109:	# Press [m] to go step 1 (new mask)
    			step = 1
                maskVertices = []
    			instructions = '[CLICK] Add vertices  -  [ENTER] Save & Finish'
            if key == 99:	# Press [c] to go step 2 (new counter)
    			step = 2
    			instructions = '[0-9] Enter ID  -  [ENTER] Save & Finish'

        # Add vertices to the mask
        elif step == 1:
    		if key == 13:
    			if len(maskVertices) == 2:
    				step = 0
    				instructions = '[m] Add mask or [c] Add counter'
    				extra = ''

    	# Counter ID
        elif step == 2:
    		if key == 13:	# Press ENTER to go step 3 (new counter name)
    			step = 3
    			instructions = '[a-z]: Enter NAME -  [ENTER] Save & Finish'
    		elif key is not 255:		# Enter numeric ID
    			counterID += chr(key%256)
    			extra = counterID

        # Counter NAME
        elif step == 3:
    		if key == 13:    # Press ENTER to go step 4 (new counter type)
    			step = 4
    			instructions = 'Enter Type: [s] SIMPLE  [o] COMPLEX -  [ENTER] Save & Finish'
    		elif key is not 255:		# Enter numeric ID
    			counterNAME += chr(key%256)
    			extra = counterNAME

        # Counter TYPE
        elif step == 4:
    		if key == 13:    # Press ENTER to go step 5 (new counter type)
    			step = 5
    			instructions = '[+] Add new lane'
                counterID = ''
                counterNAME = ''
                counterTYPE = ''
                lanes = []
    		elif key == 115:
    			cntr = counter.simpleCounter(int(counterID), str(counterNAME))
            elif key == 99:
    			cntr = counter.complexCounter(int(counterID), str(counterNAME))

    	# New lane
        elif step == 5:
    		if key == ord('+'):
                step = 6
			    instructions = '[0-9] Enter laneID  -  [ENTER] Save & Finish'

        # Lane ID
        elif step == 6:
    		if key == 13:    # Press ENTER to go step 7 (new lane name)
    			step = 7
    			instructions = '[a-z]: Enter NAME -  [ENTER] Save & Finish'
    		elif key is not 255:		# Enter numeric ID
    			laneID += chr(key%256)
    			extra = laneID

        # Lane NAME
        elif step == 7:
    		if key == 13:    # Press ENTER to go step 8 (new lane type)
    			step = 8
    			instructions = 'Enter Type: [0] INDIFERENT  [1] IN  [2] OUT  -  [ENTER] Save & Finish'
    		elif key is not 255:    # Enter numeric ID
    			laneNAME += chr(key%256)
    			extra = laneNAME

        # Lane TYPE
        elif step == 8:
    		if key == 13:    # Press ENTER to go step 9 (new lane vertices)
    			step = 9
    			instructions = '[CLICK] Add vertices  -  [ENTER] Save & Finish'
    		elif key is 48:
    			laneTYPE = 0
            elif key is 49:
    			laneTYPE = 1
            elif key is 50:
    			laneTYPE = 2

    	# Lane VERTICES
        elif step == 9:
    		if key == 13:
    			if len(laneVERTICES) == 2:
                    lane = counter.lane(int(laneID), str(laneNAME), int(laneTYPE))
                    lane.appendVertices(laneVERTICES)
                    lanes.append([lane.ID, lane.NAME, lane.TYPE, lane.VERTICES])
    				step = 5
    				instructions = '[+] Add new lane'
    				extra = ''
                    laneID = ''
                    laneNAME = ''
                    laneTYPE = ''
                    laneVERTICES = []


        if key == 115:  #Press [s] to save all geometries
            #Save mask to maskGeom.pickle
            with open('config-files/maskGeom.pickle', 'wb') as handle:
                pickle.dump(maskVertices, handle, protocol = pickle.HIGHEST_PROTOCOL )
            #Save lanes[] to counterGeom.pickle
            with open('config-files/counterGeom.pickle', 'wb') as handle:
                pickle.dump(lanes, handle, protocol = pickle.HIGHEST_PROTOCOL )


    	if key == 27:    # Press ESC to quit
    		break

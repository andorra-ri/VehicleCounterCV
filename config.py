#-----------------------------
#<-------- Libraries -------->
#-----------------------------
import cv2
import json
import os.path


#-----------------------------
#<------ Configuration ------>
#-----------------------------
instructions = '[m] Add mask or [c] Add analyzer'
extra = ''
step = 0
maskVertices = []
geomVertices = []


#-----------------------------
#<-------- Functions -------->
#-----------------------------
def loadMaskConfig(path):
    if (os.path.exists(path)):
        with open(path, 'r') as handle:
            maskConfig = json.load(handle)
            maskConfig = maskConfig["maskConfig"]
    else:
        maskConfig = {}
        maskConfig["maskConfig"] = {}
        maskConfig["maskConfig"]["vertices"] = []

    return maskConfig


def loadAnalyzerConfig(path):
    if (os.path.exists(path)):
        with open(path, 'r') as handle:
            analyzerConfig = json.load(handle)
    else:
        analyzerConfig = {}
        analyzerConfig["objectType"] = ""
        analyzerConfig["objectConfig"] = {}
        analyzerConfig["objectConfig"]["id"] = ""
        analyzerConfig["objectConfig"]["name"] = ""
        analyzerConfig["objectConfig"]["geomType"] = ""
        analyzerConfig["objectConfig"]["geomConfig"] = []

    return analyzerConfig


def drawMask(mask, img, color):
    vertices  = mask["maskConfig"]["vertices"]
    if(len(vertices > 0):
        pt1 = (vertices[0][0], vertices[0][1])
        pt2 = (vertices[1][0], vertices[1][1])

        cv2.rectangle(img, pt1, pt2, color, 3)


def drawAnalyzer(analyzer, img):
    type = analyzer["objectConfig"]["geomType"]
    geoms = analyzer["objectConfig"]["geomConfig"]
    if(type == "lane"):
        for geom in geoms:
            p1 = (geom["vertices"][0])
            p2 = (geom["vertices"][1])
            cv2.line(img, p1, p2, color, 1)
    elif(type == "zone"):
        for geom in geoms:
            p1 = (geom["vertices"][0])
            p2 = (geom["vertices"][1])
            cv2.rectangle(img, p1, p2, color, 3)



# MOUSE EVENT HANDLER
def mouseClicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if step == 1:
            if(len(maskVertices) <= 2):
                maskVertices.append([x,y])
            elif(len(maskVertices) == 2):
                print("Mask has two vertices, you can't add another vertice!")
        if step == 9:
            if(len(geomVERTICES) <= 2):
                geomVERTICES.append([x,y])
            elif(len(geomVERTICES) ==2):
                print("Lane has two vertices, you can't add another vertice!")


#-----------------------------
#<---------- Main ----------->
#-----------------------------
if __name__ == "__main__":
    #Load video here
    cap = cv2.VideoCapture('videos/testRotonda.mp4')
    ret, img = cap.read()
    cv2.namedWindow("img", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('img', mouseClicked)

    mask = loadMaskConfig('config-files/maskConfig.json')
    analyzer = loadAnalyzerConfig('config-files/analyzerConfig.json')

    ###MAIN LOOP
    while(1):
        ret, img = cap.read()

        drawMask(mask, img, [0,0,255])
        drawAnalyzer(analyzer, img)

        cv2.putText(img, instructions, (20,30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
        cv2.putText(img, extra, (20,55), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))

        cv2.imshow("img", img)

        key = cv2.waitKey(0) & 0xFF
        #Idle
        if step == 0:
            extra = ''
            if key == 109:     # Press [m] to go step 1 (new mask)
                step = 1
                instructions = '[CLICK] Add vertices  -  [ENTER] Save & Finish'
            if key == 99:      # Press [c] to go step 2 (new counter)
                step = 2
                instructions = '[a-z]: Enter TYPE  -  [ENTER] Save & Finish'

        # Add vertices to the mask
        elif step == 1:
            if key == 13:
                if len(maskVertices) == 2:
                    mask["maskConfig"]["vertices"] = maskVertices
                    step = 0
                    instructions = '[m] Add mask or [c] Add counter'
                    extra = ''
                    maskVertices = []

        # Analyzer TYPE
        elif step ==2:
            if key == 13:
                step = 3
                instructions = '[a-z]: Enter ID - [ENTER] Save & Finish'
                extra =''
            elif key is not 255:
                analyzerTYPE += chr(key%256)
                extra = analyzerTYPE
                analyzer["objectType"] = analyzerType

        # Analyzer ID
        elif step == 3:
            if key == 13:	# Press ENTER to go step 3 (new counter name)
                step = 4
                instructions = '[a-z]: Enter NAME -  [ENTER] Save & Finish'
                extra = ''
            elif key is not 255:		# Enter numeric ID
                analyzerID += chr(key%256)
                extra = analyzerID
                analyzer["objectConfig"]["id"] = analyzerID

        # Analyzer NAME
        elif step == 4:
            if key == 13:    # Press ENTER to go step 4 (new counter type)
                step = 5
                instructions = 'Enter Type: [0] SIMPLE  [1] COMPLEX -  [ENTER] Save & Finish'
                extra = ''
            elif key is not 255:		# Enter numeric ID
                analyzerNAME += chr(key%256)
                extra = analyzerNAME
                analyzer["objectConfig"]["name"] = analyzerNAME

        # Analyzer Geom TYPE
        elif step == 5:
            if key == 13:    # Press ENTER to go step 5 (new counter type)
                step = 6
                instructions = '[+] Add new geom'
                extra = ''
            elif key is not 255:
                analyzerGeomType += chr(key%256)
                extra = analyzerGeomType
                analyzer["objectConfig"]["geomType"] = analyzerGeomType

        # New geom
        elif step == 6:
            if key == ord('+'):
                step = 7
                instructions = '[0-9] Enter ID  -  [ENTER] Save & Finish'
                geom = {}

        # Geom ID
        elif step == 7:
            if key == 13:    # Press ENTER to go step 7 (new lane name)
                step = 8
                instructions = '[a-z]: Enter NAME -  [ENTER] Save & Finish'
                extra = ''
            elif key is not 255:		# Enter numeric ID
                geomID += chr(key%256)
                extra = geomID
                geom["id"] = geomID

        # Geom NAME
        elif step == 8:
            if key == 13:    # Press ENTER to go step 8 (new geom type)
                step = 9
                instructions = 'Enter Type: [0] INDIFERENT  [1] IN  [2] OUT  -  [ENTER] Save & Finish'
                extra = ''
            elif key is not 255:    # Enter numeric ID
                geomNAME += chr(key%256)
                extra = geomNAME
                geom["name"] = geomNAME

        # Geom TYPE
        elif step == 9:
            if key == 13:    # Press ENTER to go step 9 (new geom vertices)
                step = 10
                instructions = '[CLICK] Add vertices  -  [ENTER] Save & Finish'
                geom["type"] = geomType
            elif key is 48:
                geomType = 0
            elif key is 49:
                geomType = 1
            elif key is 50:
                geomType = 2

        # Geom VERTICES
        elif step == 10:
            if key == 13:
                if len(geomVERTICES) == 2:
                    geom["vertices"] = geomVERTICES
                    analyzer["objectConfig"]["geomConfig"].append(geom)
                    step = 6
                    instructions = '[+] Add new lane'
                    extra = ''
                    geomID = ''
                    geomNAME = ''
                    geomType = ''
                    geomVERTICES = []


        if key == 115:  #Press [s] to save all geometries
            #Save mask to maskConfig.json
            with open('config-files/maskConfig.json', 'w') as han:
                json.dump(mask, han)
            #Save analyzer to analyzerConfig.json
            with open('config-files/analyzerConfig.json', 'w') as handle:
                json.dump(analyzer, handle)


        if key == 27:    # Press ESC to quit
            break

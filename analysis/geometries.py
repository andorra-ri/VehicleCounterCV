import cv2

class Lane:
    def __init__(self, id, name, type, vertices):
        self.ID = id
        self.NAME = name
        self.VERTICES = vertices              #[[x1, y1], [x2, y2]]
        self.TYPE = type                      #type must be 0(indiferent), 1(in), 2(out)


    def setVertices(self, vertices):
        self.VERTICES = vertices


    def getVertices(self):
        return self.VERTICES


    def draw(self, img, color):
        cv2.line(img, tuple(self.VERTICES[0]), tuple(self.VERTICES[1]), color, 1)


class Zone:
    def __init__(self, id, name, type, vertices):
        self.ID = id
        self.NAME = name
        self.VERTICES = vertices               #[[x1, y1], [x2, y2]]
        self.TYPE = type                       #type must be 0(indiferent), 1(in), 2(out)


    def setVertices(self, vertices):
        self.VERTICES = vertices


    def getVertices(self):
        return self.VERTICES


    def draw(self, img, color):
        cv2.rectangle(img, tuple(self.VERTICES[0]), tuple(self.VERTICES[1]), color, 1)

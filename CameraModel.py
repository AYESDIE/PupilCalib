import cv2

class CameraModel():
    def __init__(self):
        pass

    def capture(self):
        cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read() # return a single frame in variable `frame`
        return ret, frame


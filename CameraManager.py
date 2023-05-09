import cv2
import warnings

class CameraManager():
    def __init__(self):
        pass

    def captureFrame(self):
        cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read() # return a single frame in variable `frame`
        if not ret:
            warnings.warn("Could not capture image", ResourceWarning)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

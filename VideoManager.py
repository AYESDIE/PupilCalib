import CameraManager
import cv2
import os


class VideoManager(CameraManager.CameraManager):
    def __init__(self, path):
        super().__init__()
        self.s_manager_name = "VideoManager"

        self.cap = cv2.VideoCapture(path)

    def captureCurrentFrame(self):
        ret, m_current_frame = self.cap.read()


        if ret:
            self.m_current_frame = cv2.cvtColor(m_current_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', self.m_current_frame)
            # cv2.waitKey(1)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.detectAndShowAprilTag()
        self.applyCalibrationMatrix()

class CoreVideoManager(CameraManager.CoreManager):
    def __init__(self, path):
        super().__init__()
        self.s_manager_name = "CoreVideoManager"

        self.cap = cv2.VideoCapture(path)

    def captureCurrentFrame(self):
        ret, m_current_frame = self.cap.read()

        if ret:
            self.m_current_frame = cv2.cvtColor(m_current_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', self.m_current_frame)
            # cv2.waitKey(1)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.detectAndShowAprilTag()
        self.applyCalibrationMatrix()

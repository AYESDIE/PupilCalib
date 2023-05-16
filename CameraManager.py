import cv2
import warnings

import numpy
from PyQt5.QtCore import *


class CameraManager():
    def __init__(self):
        self.m_current_frame = None
        self.m_camera_matrix = None
        self.m_new_camera_matrix = None
        self.m_distortion_coefficient = None

        self.b_is_applying_calibration = False

        self.m_object_points = []
        self.m_image_points = []
        self.m_object_point = numpy.zeros((9 * 6, 3), numpy.float32)
        self.m_object_point[:, :2] = numpy.mgrid[0:9, 0:6].T.reshape(-1, 2)

        self.calibation_frame_count = 0
        self.minimum_calibration_frame = 30
        self.qt_calibrate_timer_cooldown = 500
        self.qt_calibrate_timer = QTimer()
        self.qt_calibrate_timer.setSingleShot(True)
        self.qt_calibrate_timer.timeout.connect(self.calibrateCamera)

    def calibrateCamera(self):
        try:
            ret, corners = cv2.findChessboardCorners(self.current_frame, (9, 6), None)

            if ret:
                self.calibation_frame_count = self.calibation_frame_count + 1
                self.m_object_points.append(self.m_object_point)
                corners2 = cv2.cornerSubPix(self.m_current_frame, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.m_image_points.append(corners2)

                cv2.drawChessboardCorners(self.m_current_frame, (9, 6), corners, ret)

        except:
            pass

        if self.calibation_frame_count == self.minimum_calibration_frame:
            self.calibation_frame_count = 0

            ret, self.m_camera_matrix, self.m_distortion_coefficient, rvecs, tvecs = cv2.calibrateCamera(
                self.m_object_points,
                self.m_image_points,
                self.m_current_frame.shape[::-1],
                None,
                None
            )

            self.m_new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.m_camera_matrix,
                                                                          self.m_distortion_coefficient,
                                                                          (self.m_current_frame.shape[1],
                                                                           self.m_current_frame.shape[0]),
                                                                          1,
                                                                          (self.m_current_frame.shape[1],
                                                                           self.m_current_frame.shape[0])
                                                                          )

            self.m_object_points = []
            self.m_image_points = []

            return
        else:
            self.qt_calibrate_timer.start(self.qt_calibrate_timer_cooldown)
            return

    def captureCurrentFrame(self):
        cap = cv2.VideoCapture(1)  # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read()  # return a single frame in variable `frame`
        if not ret:
            warnings.warn("Could not capture image", ResourceWarning)
        self.m_current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.applyCalibrationMatrix()

    def applyCalibrationMatrix(self):
        if self.b_is_applying_calibration:
            try:
                self.calibrate_frame = cv2.undistort(self.m_current_frame, self.m_camera_matrix, self.m_distortion_coefficient, None, self.m_new_camera_matrix)
            except:
                warnings.warn("CameraManager::applyCalibrationMatrix - Please apply calibration first")
        return

    def getCurrentFrame(self):
        return self.m_current_frame

    def setCameraMatrix(self, camera_matrix):
        self.m_camera_matrix = camera_matrix

    def setCalibration(self, calibration: bool):
        self.b_is_applying_calibration = calibration


class CoreManager():
    def __init__(self):
        pass

    def captureFrame(self):
        cap = cv2.VideoCapture(1)  # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read()  # return a single frame in variable `frame`
        if not ret:
            warnings.warn("Could not capture image", ResourceWarning)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return [frame, frame, frame]

import cv2
import warnings

import numpy
from PyQt5.QtCore import *

import pupil_apriltags


class CameraManager():
    def __init__(self):
        self.m_current_frame = None
        self.m_camera_matrix = None
        self.m_new_camera_matrix = None
        self.m_distortion_coefficient = None
        self.m_roi = None

        self.b_is_applying_calibration = False
        self.b_is_applying_april_detection = False

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

        self.detector = pupil_apriltags.Detector("tag36h11")

        self.s_manager_name = "CameraManager"

        self.world_origin_coords = [[None, None], [None, None]]
        self.b_world_complete = False

        self.r_vec = None
        self.t_vec = None
        self.r_mat = None

        self.proj_mat = None

        self.camera_3d_point_place = None
        self.image_p = None

        self.test_point = None

    def calibrateCamera(self):
        print(f"{self.s_manager_name}::calibrateCamera - step: {self.calibation_frame_count}")
        try:
            ret, corners = cv2.findChessboardCorners(self.m_current_frame, (9, 6), None)

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

            self.m_new_camera_matrix, self.m_roi = cv2.getOptimalNewCameraMatrix(self.m_camera_matrix,
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
        self.detectAndShowAprilTag()

    def getCurrentFrame(self):
        return self.m_current_frame

    def setCameraMatrix(self, camera_matrix):
        self.m_camera_matrix = camera_matrix

    def setCalibration(self, calibration: bool):
        self.b_is_applying_calibration = calibration
        print(f"{self.s_manager_name}::setCalibration - Calibration is now {self.b_is_applying_calibration}")

    def saveCameraCalibration(self):
        if self.m_new_camera_matrix is not None and self.m_camera_matrix is not None and self.m_distortion_coefficient is not None:
            numpy.save(self.s_manager_name + "_camera_matrix.npy", self.m_camera_matrix, allow_pickle=True)
            numpy.save(self.s_manager_name + "_new_camera_matrix.npy", self.m_new_camera_matrix, allow_pickle=True)
            numpy.save(self.s_manager_name + "_distortion_coefficient.npy", self.m_distortion_coefficient,
                       allow_pickle=True)
            numpy.save(self.s_manager_name + "_roi.npy", self.m_roi, allow_pickle=True)
            print(f"{self.s_manager_name}::saveCameraCalibration - Saved successfully.")
            return True
        else:
            print(f"{self.s_manager_name}::saveCameraCalibration - Calibrate the camera before saving.")
            return False

    def loadCameraCalibration(self):
        try:
            self.m_camera_matrix = numpy.load(self.s_manager_name + "_camera_matrix.npy")
            self.m_new_camera_matrix = numpy.load(self.s_manager_name + "_new_camera_matrix.npy")
            self.m_distortion_coefficient = numpy.load(self.s_manager_name + "_distortion_coefficient.npy")
            self.m_roi = numpy.load(self.s_manager_name + "_roi.npy")
            print(f"{self.s_manager_name}::loadCameraCalibration - Loaded successfully.")
            print(f"{self.s_manager_name}::loadCameraCalibration - Camera Matrix:\n{self.m_camera_matrix}")
            print(f"{self.s_manager_name}::loadCameraCalibration - New Camera Matrix:\n{self.m_new_camera_matrix}")
            return True
        except:
            print(f"{self.s_manager_name}::loadCameraCalibration - Failed to load Calibration Matrix")
            return False

    def setAprilDetection(self, april : bool):
        self.b_is_applying_april_detection = april
        print(f"{self.s_manager_name}::setAprilDetection - Detection is now {self.b_is_applying_april_detection}")

    def applyCalibrationMatrix(self):
        if self.b_is_applying_calibration:
            self.m_current_frame = cv2.undistort(self.m_current_frame, self.m_camera_matrix,
                                                 self.m_distortion_coefficient, None, self.m_new_camera_matrix)
            x, y, w, h = self.m_roi
            self.m_current_frame = cv2.resize(self.m_current_frame[y:y + h, x:x + w],
                                              (self.m_current_frame.shape[1], self.m_current_frame.shape[0]),
                                              cv2.INTER_AREA)

    def detectAndShowAprilTag(self):
        # mcc = cv2.cvtColor(self.m_current_frame, cv2.COLOR_GRAY2RGB)
        if self.b_is_applying_april_detection:
            detection = self.detector.detect(self.m_current_frame)

            self.world_origin_coords = numpy.zeros([5, 2], dtype=numpy.float32)
            self.b_world_complete = False

            cv2.circle(self.m_current_frame, (int(50), int(100)),
                       radius=10,
                       color=(255, 255, 255),
                       thickness=3
                       )

            if self.image_p is not None:
                try:
                    cv2.circle(self.m_current_frame, (int(self.image_p[0]), int(self.image_p[1])),
                               radius=10,
                               color=(255, 255, 255),
                               thickness=3
                               )
                except:
                    pass

            for result in detection:
                if result.tag_id == 4:
                    self.test_point = result.center
                    continue
                (ptA, ptB, ptC, ptD) = result.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))
                # draw the bounding box of the AprilTag detection
                cv2.line(self.m_current_frame, ptA, ptC, (255, 255, 255), 10)
                cv2.line(self.m_current_frame, ptB, ptD, (255, 255, 255), 10)

                if len(detection) == 5:
                    self.world_origin_coords[result.tag_id] = result.center
                    

            if len(detection) == 5:
                world_object = numpy.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 0]
                ], dtype=numpy.float32)

                retval, self.r_vec, self.t_vec = cv2.solvePnP(world_object, self.world_origin_coords[0:4], self.m_camera_matrix, self.m_distortion_coefficient)
                self.r_mat = numpy.array(cv2.Rodrigues(self.r_vec)[0])

                proj_mat = numpy.concatenate((self.r_mat, self.t_vec.reshape(3, 1)), axis = 1)
                proj_mat = numpy.concatenate((proj_mat, numpy.zeros([1, 4])))
                proj_mat[3, 3] = 1.
                self.proj_mat = proj_mat

    def setTestCoords(self, coords):
        if self.proj_mat is not None and coords is not None:
            # print(coords)
            # self.camera_3d_point_place = numpy.matmul(self.proj_mat, coords)[:3]
            self.image_p = cv2.projectPoints(coords, self.r_vec, self.t_vec, self.m_camera_matrix, self.m_distortion_coefficient)[0][0][0]


    def calculate_XYZ(self, tp):
        if tp is None:
            return None

        u = tp[0]
        v = tp[1]
        scaling_factor = 1.75

        uv_1 = numpy.array([[u, v, 1]], dtype = numpy.float32)
        uv_1 = uv_1.T

        suv_1 = scaling_factor * uv_1
        xyz_c = numpy.linalg.inv(self.m_new_camera_matrix).dot(suv_1)
        xyz_c = xyz_c - self.t_vec
        XYZ = numpy.linalg.inv(self.r_mat).dot(xyz_c)

        return XYZ

class CoreManager(CameraManager):
    def __init__(self):
        super(CoreManager, self).__init__()
        self.s_manager_name = "CoreManager"
        self.m_current_left = numpy.zeros([10, 10])
        self.m_current_right = numpy.zeros([10, 10])
        pass

    def captureCurrentFrame(self):
        cap = cv2.VideoCapture(1)  # video capture source camera (Here webcam of laptop)
        ret, frame = cap.read()  # return a single frame in variable `frame`
        if not ret:
            warnings.warn("Could not capture image", ResourceWarning)
        self.m_current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.m_current_left = self.m_current_frame
        self.m_current_right = self.m_current_frame

        self.applyCalibrationMatrix()
        self.detectAndShowAprilTag()

    def getCurrentFrame(self):
        return [self.m_current_frame, self.m_current_right, self.m_current_left]

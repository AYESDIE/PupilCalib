from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import numpy
import sys


class PupilCalibManager(QWidget):
    SIGNAL_update_frame = pyqtSignal()

    def __init__(self, camera_manager, pupil_manager):
        super(PupilCalibManager, self).__init__()

        #self.output_layout = QVBoxLayout()
        self.main_layout = QGridLayout()
        self.output_graphics_view = QGraphicsView()

        self.world_camera_layout = QVBoxLayout()
        self.pupil_camera_layout = QVBoxLayout()

        # main camera stuff
        self.camera_manager = camera_manager
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.newcameramtx = None
        self.current_frame = None
        self.current_width = None
        self.current_height = None
        self.calibrate_frame = None
        self.calibrate_image = QLabel()
        self.calibrate_image.setScaledContents(False)
        self.calibrate_width = None
        self.calibrate_height = None
        self.b_calibrate_world_camera = False
        self.current_image = QLabel()
        self.current_image.setScaledContents(False)

        # pupil cam stuff
        self.pupil_manager = pupil_manager
        self.pupil_world_frame = None
        self.pupil_world_image = QLabel()

        self.pupil_eye_left_frame = None
        self.pupil_eye_left_image = QLabel()

        self.pupil_eye_right_frame = None
        self.pupil_eye_right_image = QLabel()

        self.objp = numpy.zeros((9 * 6, 3), numpy.float32)
        self.objp[:, :2] = numpy.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

        self.showMaximized()
        self.raise_()

        self.setLayout(self.main_layout)

        self.update_frame_timer = QTimer()
        self.update_frame_timer.start(100)

        self.button_cooldown_timer = QTimer()
        self.button_cooldown_timer.setSingleShot(True)
        self.button_cooldown_timer.timeout.connect(self.stopWorldCalibration)

        self.button = QPushButton("Chessboard Calibrate")
        self.button.clicked.connect(self.onButtonClick)

        self.update_frame_timer.timeout.connect(self.updateFrame)

        # self.camera_feed_layout.addWidget(self.current_image, 1, 1)
        # self.camera_feed_layout.addWidget(self.calibrate_image, 1, 2)
        # self.camera_feed_layout.addWidget(self.pupil_world_image, 2, 1)
        # self.camera_feed_layout.addWidget(self.pupil_eye_left_image, 2, 2)
        # self.camera_feed_layout.addWidget(self.pupil_eye_right_image, 2, 3)
        # self.output_layout.addLayout(self.camera_feed_layout, 90)
        # self.output_layout.addWidget(self.button, 10)
        self.pupil_camera_layout.addWidget(self.pupil_world_image)
        self.pupil_camera_layout.addWidget(self.pupil_eye_left_image)
        self.pupil_camera_layout.addWidget(self.pupil_eye_right_image)

        self.world_camera_layout.addWidget(self.current_image, 46)
        self.world_camera_layout.addWidget(self.calibrate_image, 46)
        self.world_camera_layout.addWidget(self.button, 8)

        self.main_layout.addLayout(self.pupil_camera_layout, 1, 1)
        self.main_layout.addLayout(self.world_camera_layout, 1, 2)


    def onButtonClick(self):
        self.b_calibrate_world_camera = True
        if self.b_calibrate_world_camera:
            self.button_cooldown_timer.start(2000)
            self.button.setEnabled(False)

        print(f"onButtonClick(): calibration: {self.b_calibrate_world_camera}")

    def stopWorldCalibration(self):
        self.b_calibrate_world_camera = False
        self.button.setEnabled(True)

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                                               self.current_frame.shape[::-1], None,
                                                                               None)
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist,
                                                               (self.current_frame.shape[1],
                                                                self.current_frame.shape[0]), 1,
                                                               (self.current_frame.shape[1],
                                                                self.current_frame.shape[0]))

        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]

        self.objp = numpy.zeros((9 * 6, 3), numpy.float32)
        self.objp[:, :2] = numpy.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

    def updateFrame(self):
        self.current_frame = self.camera_manager.captureFrame()
        self.current_image.setPixmap(QPixmap(QImage(self.current_frame,
                                                    self.current_frame.shape[1],
                                                    self.current_frame.shape[0],
                                                    QImage.Format_Grayscale8)))
        # Calibrate stuff
        self.calibrateWorldCamera()
        self.applyWorldCameraCalibration()

        self.pupil_world_frame, self.pupil_eye_left_frame, self.pupil_eye_right_frame = self.pupil_manager.capture()
        self.pupil_world_image.setPixmap(QPixmap(QImage(self.pupil_world_frame,
                                                        self.pupil_world_frame.shape[1],
                                                        self.pupil_world_frame.shape[0],
                                                        QImage.Format_Grayscale8)))

        self.pupil_eye_left_image.setPixmap(QPixmap(QImage(self.pupil_eye_left_frame,
                                                           self.pupil_eye_left_frame.shape[1],
                                                           self.pupil_eye_left_frame.shape[0],
                                                           QImage.Format_Grayscale8)))

        self.pupil_eye_right_image.setPixmap(QPixmap(QImage(self.pupil_eye_right_frame,
                                                            self.pupil_eye_right_frame.shape[1],
                                                            self.pupil_eye_right_frame.shape[0],
                                                            QImage.Format_Grayscale8)))

    def applyWorldCameraCalibration(self):
        try:
            self.calibrate_frame = cv2.undistort(self.current_frame, self.mtx, self.dist, None, self.newcameramtx)
            self.calibrate_image.setPixmap(QPixmap(QImage(self.calibrate_frame,
                                                          self.calibrate_frame.shape[1],
                                                          self.calibrate_frame.shape[0],
                                                          QImage.Format_Grayscale8)))
        except:
            pass

    def calibrateWorldCamera(self):
        if self.b_calibrate_world_camera:
            ret, corners = cv2.findChessboardCorners(self.current_frame, (9, 6), None)

            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(self.current_frame, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.imgpoints.append(corners2)

                cv2.drawChessboardCorners(self.current_frame, (9, 6), corners, ret)

            else:
                self.stopWorldCalibration()

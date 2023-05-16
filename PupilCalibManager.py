from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import numpy
import sys
#
# import apriltag


class PupilCalibManager(QWidget):
    SIGNAL_update_frame = pyqtSignal()

    def __init__(self, camera_manager, pupil_manager):
        super(PupilCalibManager, self).__init__()

        #self.output_layout = QVBoxLayout()
        self.main_layout = QGridLayout()
        self.output_graphics_view = QGraphicsView()

        self.world_camera_layout = QVBoxLayout()
        self.pupil_camera_layout = QVBoxLayout()
        self.pupil_eye_layout = QHBoxLayout()

        # main camera stuff
        self.camera_manager = camera_manager
        self.current_frame = None
        self.current_width = None
        self.current_height = None
        self.calibrate_frame = numpy.zeros((4, 4))
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


        self.update_frame_timer = QTimer()
        self.update_frame_timer.start(1)
        self.update_frame_timer.timeout.connect(self.updateFrame)

        self.world_calibrate_button = QPushButton("World Calibrate")
        self.world_calibrate_button.clicked.connect(self.onWorldCalibrateButtonClick)
        self.world_save_button = QPushButton("Save Calibration")
        self.world_save_button.clicked.connect(self.onWorldSaveButtonClick)
        self.world_load_button = QPushButton("Load Calibration")
        self.world_load_button.clicked.connect(self.onWorldLoadButtonClick)
        self.world_apply_button = QPushButton("Apply Calibration")
        self.world_apply_button.clicked.connect(self.onWorldApplyButtonClick)
        self.world_camera_layout.addWidget(self.current_image, 60)
        self.world_camera_layout.addWidget(self.world_calibrate_button, 10)
        self.world_camera_layout.addWidget(self.world_save_button, 10)
        self.world_camera_layout.addWidget(self.world_load_button, 10)
        self.world_camera_layout.addWidget(self.world_apply_button, 10)


        self.pupil_camera_layout.addWidget(self.pupil_world_image)
        self.pupil_eye_layout.addWidget(self.pupil_eye_left_image)
        self.pupil_eye_layout.addWidget(self.pupil_eye_right_image)
        self.pupil_camera_layout.addLayout(self.pupil_eye_layout)


        self.main_layout.addLayout(self.pupil_camera_layout, 1, 1)
        self.main_layout.addLayout(self.world_camera_layout, 1, 2)


        self.setLayout(self.main_layout)

        self.showMaximized()
        self.raise_()


    def onWorldCalibrateButtonClick(self):
        self.camera_manager.calibrateCamera()
        print(f"onButtonClick(): calibration: {self.b_calibrate_world_camera}")

    def onWorldSaveButtonClick(self):
        self.camera_manager.saveCameraCalibration()

    def onWorldLoadButtonClick(self):
        self.camera_manager.loadCameraCalibration()

    def onWorldApplyButtonClick(self):
        self.camera_manager.setCalibration(not self.camera_manager.b_is_applying_calibration)

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
        self.camera_manager.captureCurrentFrame()
        self.current_frame = self.camera_manager.getCurrentFrame()
        self.current_image.setPixmap(QPixmap(QImage(self.current_frame,
                                                    self.current_frame.shape[1],
                                                    self.current_frame.shape[0],
                                                    QImage.Format_Grayscale8)))
        # Calibrate stuff
        # self.calibrateWorldCamera()
        # self.applyWorldCameraCalibration()
        # self.pupil_manager.recent_world = self.calibrate_frame
        # self.pupil_manager.applyApril()
        # self.pupil_manager.cv2()

        self.pupil_world_frame, self.pupil_eye_left_frame, self.pupil_eye_right_frame = self.pupil_manager.captureFrame()
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
    #
    # def applyWorldCameraCalibration(self):
    #     try:
    #         self.calibrate_frame = cv2.undistort(self.current_frame, self.mtx, self.dist, None, self.newcameramtx)
    #         self.calibrate_image.setPixmap(QPixmap(QImage(self.calibrate_frame,
    #                                                       self.calibrate_frame.shape[1],
    #                                                       self.calibrate_frame.shape[0],
    #                                                       QImage.Format_Grayscale8)))
    #
    #     except:
    #         pass
    #
    # def calibrateWorldCamera(self):
    #     if self.b_calibrate_world_camera:
    #         try:
    #             ret, corners = cv2.findChessboardCorners(self.current_frame, (9, 6), None)
    #
    #             if ret:
    #                 self.world_calib_frames = self.world_calib_frames + 1
    #                 self.objpoints.append(self.objp)
    #                 corners2 = cv2.cornerSubPix(self.current_frame, corners, (11, 11), (-1, -1),
    #                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    #                 self.imgpoints.append(corners2)
    #
    #                 cv2.drawChessboardCorners(self.current_frame, (9, 6), corners, ret)
    #
    #         except:
    #             pass
    #
    #     if self.world_calib_frames == 30:
    #         self.world_calib_frames = 0
    #         self.stopWorldCalibration()
    #         return
    #     else:
    #         self.button_cooldown_timer.start(500)

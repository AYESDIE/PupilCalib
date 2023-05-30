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
        self.pupil_world_image = QLabel()
        self.pupil_eye_left_image = QLabel()
        self.pupil_eye_right_image = QLabel()


        self.update_frame_timer = QTimer()
        self.update_frame_timer.start(10)
        self.update_frame_timer.timeout.connect(self.updateFrame)

        self.world_calibrate_button = QPushButton("World Calibrate")
        self.world_calibrate_button.clicked.connect(self.onWorldCalibrateButtonClick)
        self.world_save_button = QPushButton("Save Calibration")
        self.world_save_button.clicked.connect(self.onWorldSaveButtonClick)
        self.world_load_button = QPushButton("Load Calibration")
        self.world_load_button.clicked.connect(self.onWorldLoadButtonClick)
        self.world_apply_button = QPushButton("Apply Calibration")
        self.world_apply_button.clicked.connect(self.onWorldApplyButtonClick)
        self.world_april_button = QPushButton("World April")
        self.world_april_button.clicked.connect(self.onWorldAprilButtonClick)
        self.world_camera_layout.addWidget(self.current_image, 60)
        self.world_camera_layout.addWidget(self.world_calibrate_button, 10)
        self.world_camera_layout.addWidget(self.world_save_button, 10)
        self.world_camera_layout.addWidget(self.world_load_button, 10)
        self.world_camera_layout.addWidget(self.world_apply_button, 10)
        self.world_camera_layout.addWidget(self.world_april_button, 10)


        self.pupil_calibrate_button = QPushButton("Pupil Calibrate")
        self.pupil_calibrate_button.clicked.connect(self.onPupilCalibrateButtonClick)
        self.pupil_save_button = QPushButton("Save Calibration")
        self.pupil_save_button.clicked.connect(self.onPupilSaveButtonClick)
        self.pupil_load_button = QPushButton("Load Calibration")
        self.pupil_load_button.clicked.connect(self.onPupilLoadButtonClick)
        self.pupil_apply_button = QPushButton("Apply Calibration")
        self.pupil_apply_button.clicked.connect(self.onPupilApplyButtonClick)
        self.pupil_april_button = QPushButton("Pupil April")
        self.pupil_april_button.clicked.connect(self.onPupilAprilButtonClick)
        
        self.pupil_camera_layout.addWidget(self.pupil_world_image)
        self.pupil_eye_layout.addWidget(self.pupil_eye_left_image)
        self.pupil_eye_layout.addWidget(self.pupil_eye_right_image)
        self.pupil_camera_layout.addLayout(self.pupil_eye_layout)
        self.pupil_camera_layout.addWidget(self.pupil_calibrate_button, 10)
        self.pupil_camera_layout.addWidget(self.pupil_save_button, 10)
        self.pupil_camera_layout.addWidget(self.pupil_load_button, 10)
        self.pupil_camera_layout.addWidget(self.pupil_apply_button, 10)
        self.pupil_camera_layout.addWidget(self.pupil_april_button, 10)


        self.main_layout.addLayout(self.pupil_camera_layout, 1, 1)
        self.main_layout.addLayout(self.world_camera_layout, 1, 2)


        self.setLayout(self.main_layout)

        self.show()
        self.raise_()


    def onWorldCalibrateButtonClick(self):
        self.camera_manager.calibrateCamera()

    def onWorldSaveButtonClick(self):
        self.camera_manager.saveCameraCalibration()

    def onWorldLoadButtonClick(self):
        self.camera_manager.loadCameraCalibration()

    def onWorldApplyButtonClick(self):
        self.camera_manager.setCalibration(not self.camera_manager.b_is_applying_calibration)

    def onWorldAprilButtonClick(self):
        self.camera_manager.setAprilDetection(not self.camera_manager.b_is_applying_april_detection)

    def onPupilCalibrateButtonClick(self):
        self.pupil_manager.calibrateCamera()

    def onPupilSaveButtonClick(self):
        self.pupil_manager.saveCameraCalibration()

    def onPupilLoadButtonClick(self):
        self.pupil_manager.loadCameraCalibration()

    def onPupilApplyButtonClick(self):
        self.pupil_manager.setCalibration(not self.pupil_manager.b_is_applying_calibration)

    def onPupilAprilButtonClick(self):
        self.pupil_manager.setAprilDetection(not self.pupil_manager.b_is_applying_april_detection)

    def updateFrame(self):
        self.camera_manager.captureCurrentFrame()
        current_frame = self.camera_manager.getCurrentFrame()
        self.current_image.setPixmap(QPixmap(QImage(current_frame,
                                                    current_frame.shape[1],
                                                    current_frame.shape[0],
                                                    QImage.Format_Grayscale8)))

        self.pupil_manager.captureCurrentFrame()
        pupil_world_frame, pupil_eye_left_frame, pupil_eye_right_frame = self.pupil_manager.getCurrentFrame()
        self.pupil_world_image.setPixmap(QPixmap(QImage(pupil_world_frame,
                                                        pupil_world_frame.shape[1],
                                                        pupil_world_frame.shape[0],
                                                        QImage.Format_Grayscale8)))

        self.pupil_eye_left_image.setPixmap(QPixmap(QImage(pupil_eye_left_frame,
                                                           pupil_eye_left_frame.shape[1],
                                                           pupil_eye_left_frame.shape[0],
                                                           QImage.Format_Grayscale8)))

        self.pupil_eye_right_image.setPixmap(QPixmap(QImage(pupil_eye_right_frame,
                                                            pupil_eye_right_frame.shape[1],
                                                            pupil_eye_right_frame.shape[0],
                                                            QImage.Format_Grayscale8)))
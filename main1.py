from PyQt5.QtWidgets import *
from PupilCalibManager import PupilCalibManager
from CameraManager import CameraManager
from IDSManager import IDSManager
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ = PupilCalibManager(IDSManager())
    sys.exit(app.exec_())
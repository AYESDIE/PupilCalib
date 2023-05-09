from PyQt5.QtWidgets import *
from PupilCalibManager import PupilCalibManager
from CameraManager import CameraManager, CoreManager
from IDSManager import IDSManager
from PupilCoreManager import PupilCoreManager
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ = PupilCalibManager(IDSManager(), PupilCoreManager())
    sys.exit(app.exec_())
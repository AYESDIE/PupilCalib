from PyQt5.QtWidgets import *
from PupilCalibManager import PupilCalibManager
from CameraManager import CameraManager, CoreManager
from IDSManager import IDSManager
from PupilCoreManager import PupilCoreManager
import sys

from VideoManager import VideoManager, CoreVideoManager

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ = PupilCalibManager(VideoManager('assets/world_cam_april.MP4'), CoreVideoManager('assets/scene_cam_april.MP4'))
    sys.exit(app.exec_())
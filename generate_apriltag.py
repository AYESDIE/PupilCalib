#!/usr/bin/env python3
from moms_apriltag import ApriltagBoard
#import imageio

import cv2

board = ApriltagBoard.create(4,6,"tag25h9", 0.02)
tgt = board.board

# filename = "apriltag_target.png"
cv2.imwrite("april-tag.jpg", tgt)
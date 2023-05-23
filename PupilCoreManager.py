"""
Receive world camera data from Pupil using ZMQ.
Make sure the frame publisher plugin is loaded and confugured to gray or rgb
"""
import zmq
from msgpack import unpackb, packb, loads
import numpy
import cv2
import warnings

from PyQt5.QtCore import QThread
import time
import threading

from CameraManager import CoreManager

class PupilCoreManager(CoreManager):
    def __init__(self):
        super(PupilCoreManager, self).__init__()
        self.s_manager_name = "PupilCoreManager"

        self.s_calibration_file = "PupilCore"

        self.context = zmq.Context()
        # open a req port to talk to pupil
        addr = "127.0.0.1"  # remote ip or localhost
        req_port = "50020"  # same as in the pupil remote gui
        self.req = self.context.socket(zmq.REQ)
        self.req.connect("tcp://{}:{}".format(addr, req_port))
        # ask for the sub port
        self.req.send_string("SUB_PORT")
        warnings.warn("Please make sure Pupil Capture application is running", ResourceWarning)
        sub_port = self.req.recv_string()

        # open a sub port to listen to pupil
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect("tcp://{}:{}".format(addr, sub_port))
        self.sub.setsockopt(zmq.LINGER, 1)
        #self.sub.setsockopt(zmq.RCVHWM, 10)
        self.sub.setsockopt(zmq.RCVHWM, 20)
        self.sub.setsockopt(zmq.SNDHWM, 20)
        #self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "frame.")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "gaze")

        # set subscriptions to topics
        # recv just pupil/gaze/notifications

        self.FRAME_FORMAT = "bgr"

        # Set the frame format via the Network API plugin
        self.notify({"subject": "frame_publishing.set_format", "format": self.FRAME_FORMAT})

        self.m_current_frame = numpy.zeros((10, 10))
        self.m_current_right = numpy.zeros((10, 10))
        self.m_current_left = numpy.zeros((10, 10))

        self.pupil_thread = threading.Thread(target=self.ppc_thread_worker)
        self.pupil_thread.start()

        self.loadCameraCalibration()
        self.setCalibration(True)

        self.raw_x = 0
        self.raw_y = 0
        self.confidence = 0

    def notify(self, notification):
        """Sends ``notification`` to Pupil Remote"""
        topic = "notify." + notification["subject"]
        payload = packb(notification, use_bin_type=True)
        self.req.send_string(topic, flags=zmq.SNDMORE)
        self.req.send(payload)
        return self.req.recv_string()

    def recv_from_sub(self):
        """Recv a message with topic, payload.

        Topic is a utf-8 encoded string. Returned as unicode object.
        Payload is a msgpack serialized dict. Returned as a python dict.

        Any addional message frames will be added as a list
        in the payload dict with key: '__raw_data__' .
        """
        topic = self.sub.recv_string()
        payload = None
        msg = None
        if "gaze" in topic:
            msg = self.sub.recv()
            msg = loads(msg, raw=False)
            self.raw_x, self.raw_y = msg["norm_pos"]
            self.confidence = msg["confidence"]
        else:
            payload = unpackb(self.sub.recv(), raw=False)
        extra_frames = []
        while self.sub.get(zmq.RCVMORE):
            extra_frames.append(self.sub.recv())
        if extra_frames:
            payload["__raw_data__"] = extra_frames
        return topic, payload

    def has_new_data_available(self):
        return self.sub.get(zmq.EVENTS) & zmq.POLLIN

    def qqthread(self):
        i = 1
        while True:
            print(i)
            i = i + 1

    def captureCurrentFrame(self):
        self.detectAndShowAprilTag()
        self.showGaze()
        self.applyCalibrationMatrix()

    def showGaze(self):
        # draw confidence circle
        cv2.circle(self.m_current_frame, (int(self.m_current_frame.shape[1] * self.raw_x),
                                          self.m_current_frame.shape[0] - int(
                                              self.m_current_frame.shape[0] * self.raw_y)),
                              radius = int(15 * (2 - self.confidence)),
                              color = (255, 255, 255),
                              thickness = 3
                              )

        # draw gaze circle
        cv2.circle(self.m_current_frame, (int(self.m_current_frame.shape[1] * self.raw_x),
                                          self.m_current_frame.shape[0] - int(
                                              self.m_current_frame.shape[0] * self.raw_y)),
                   radius=10,
                   color=(255, 255, 255),
                   thickness=cv2.FILLED)

    def ppc_thread_worker(self):
        while True:
            b_gotWorld = False
            b_gotLeftEye = False
            b_gotRightEye = False

            r_world = None
            r_wH = None
            r_wW = None
            r_leftEye = None
            r_rH = None
            r_rW = None
            r_rightEye = None
            r_lH = None
            r_lW = None
            while not b_gotWorld or not b_gotLeftEye or not b_gotRightEye:
                if self.has_new_data_available():
                    topic, msg = self.recv_from_sub()

                    if topic.startswith("frame.") and msg["format"] != self.FRAME_FORMAT:
                        print(
                            f"different frame format ({msg['format']}); "
                            f"skipping frame from {topic}"
                        )
                        return

                    if topic == "frame.world":
                        r_world = msg["__raw_data__"][0]
                        r_wW = msg["width"]
                        r_wH = msg["height"]
                        b_gotWorld = True


                    elif topic == "frame.eye.0":
                        r_rightEye = msg["__raw_data__"][0]
                        r_rW = msg["width"]
                        r_rH = msg["height"]
                        b_gotRightEye = True

                    elif topic == "frame.eye.1":
                        r_leftEye = msg["__raw_data__"][0]
                        r_lW = msg["width"]
                        r_lH = msg["height"]
                        b_gotLeftEye = True

            self.m_current_frame = cv2.convertScaleAbs(
                cv2.cvtColor(numpy.frombuffer(
                    r_world, dtype=numpy.uint8
                ).reshape(r_wH, r_wW, 3), cv2.COLOR_BGR2GRAY))
            self.m_current_right = cv2.convertScaleAbs(cv2.rotate(cv2.cvtColor(numpy.frombuffer(
                r_rightEye, dtype=numpy.uint8
            ).reshape(r_rH, r_rW, 3), cv2.COLOR_BGR2GRAY), cv2.ROTATE_180))
            self.m_current_left = cv2.convertScaleAbs(cv2.cvtColor(numpy.frombuffer(
                r_leftEye, dtype=numpy.uint8
            ).reshape(r_lH, r_lW, 3), cv2.COLOR_BGR2GRAY))

    def cv2(self):
        cv2.imshow("world", self.m_current_frame)
        cv2.imshow("eye_right", self.m_current_right)
        cv2.imshow("eye_left", self.m_current_left)
        cv2.waitKey(1)


if __name__=="__main__":
    ppc = PupilCoreManager()
    while True:
        ppc.cv2()
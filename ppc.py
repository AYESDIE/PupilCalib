import zmq
from msgpack import unpackb, packb
import numpy
import cv2
import warnings

from PyQt5.QtCore import QThread
import time
import threading

class PPC():
    def __init__(self):
        self.recent_world = numpy.zeros((10, 10))
        self.recent_eye0 = numpy.zeros((10, 10))
        self.recent_eye1 = numpy.zeros((10, 10))

        self.pupil_process = threading.Thread(target = self.pupil_worker)
        self.pupil_process.start()

    def pupil_worker(self):
        def notify(notification):
            """Sends ``notification`` to Pupil Remote"""
            topic = "notify." + notification["subject"]
            payload = packb(notification, use_bin_type=True)
            req.send_string(topic, flags=zmq.SNDMORE)
            req.send(payload)
            return req.recv_string()

        def recv_from_sub():
            topic = sub.recv_string()
            payload = unpackb(sub.recv(), raw=False)
            extra_frames = []
            while sub.get(zmq.RCVMORE):
                extra_frames.append(sub.recv())
            if extra_frames:
                payload["__raw_data__"] = extra_frames
            return topic, payload

        def has_new_data_available():
            return sub.get(zmq.EVENTS) & zmq.POLLIN

        def capture():
            recent_world = None
            recent_eye0 = None
            recent_eye1 = None
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
                if has_new_data_available():
                    topic, msg = recv_from_sub()

                    if topic.startswith("frame.") and msg["format"] != FRAME_FORMAT:
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

            recent_world = cv2.resize(cv2.cvtColor(numpy.frombuffer(
                r_world, dtype=numpy.uint8
            ).reshape(r_wH, r_wW, 3), cv2.COLOR_BGR2GRAY), (640, 360), cv2.INTER_AREA)
            recent_eye0 = cv2.cvtColor(numpy.frombuffer(
                r_rightEye, dtype=numpy.uint8
            ).reshape(r_rH, r_rW, 3), cv2.COLOR_BGR2GRAY)
            recent_eye1 = cv2.cvtColor(numpy.frombuffer(
                r_leftEye, dtype=numpy.uint8
            ).reshape(r_lH, r_lW, 3), cv2.COLOR_BGR2GRAY)

            return [recent_world, recent_eye0, recent_eye1]

        context = zmq.Context()
        addr = "127.0.0.1"  # remote ip or localhost
        req_port = "50020"  # same as in the pupil remote gui

        req = context.socket(zmq.REQ)
        req.connect("tcp://{}:{}".format(addr, req_port))
        # ask for the sub port
        req.send_string("SUB_PORT")

        sub_port = req.recv_string()

        # open a sub port to listen to pupil
        sub = context.socket(zmq.SUB)
        sub.connect("tcp://{}:{}".format(addr, sub_port))
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt_string(zmq.SUBSCRIBE, "frame.")

        # set subscriptions to topics
        # recv just pupil/gaze/notifications

        FRAME_FORMAT = "bgr"

        # Set the frame format via the Network API plugin
        notify({"subject": "frame_publishing.set_format", "format": FRAME_FORMAT})

        while True:
            #global Grecent_world, Grecent_eye0, Grecent_eye1
            self.recent_world, self.recent_eye0, self.recent_eye1 = capture()


if __name__ == "__main__":
    ppc = PPC()
    while True:
        print(ppc.recent_world)
        cv2.imshow("world", ppc.recent_world)

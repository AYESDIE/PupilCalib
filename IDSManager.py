import cv2
import warnings
from CameraManager import CameraManager

from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension

import numpy


class IDSManager(CameraManager):
    def __init__(self):
        super().__init__()
        ids_peak.Library.Initialize()

        # Initialize and find device
        self.device_manager = ids_peak.DeviceManager.Instance()

        self.device_manager.Update()

        if self.device_manager.Devices().empty():
            warnings.warn("No devices found, the program will not run", ResourceWarning)

        self.device = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.node_map_remote_device = self.device.RemoteDevice().NodeMaps()[0]

        # Start data stream
        self.data_stream = self.device.DataStreams()
        if self.device.DataStreams().empty():
            warnings.warn("Data stream is not available", ResourceWarning)

        self.data_stream = self.device.DataStreams()[0].OpenDataStream()

        # TODO: Set ROI

        # Clear buffers
        self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in self.data_stream.AnnouncedBuffers():
            self.data_stream.RevokeBuffer(buffer)

        payload_size = self.node_map_remote_device.FindNode("PayloadSize").Value()
        num_buffers_min_required = self.data_stream.NumBuffersAnnouncedMinRequired()

        for count in range(num_buffers_min_required):
            buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buffer)

        # Start Acquisition

        self.data_stream.StartAcquisition(ids_peak.AcquisitionStartMode_Default, ids_peak.DataStream.INFINITE_NUMBER)
        self.node_map_remote_device.FindNode("TLParamsLocked").SetValue(1)
        self.node_map_remote_device.FindNode("AcquisitionStart").Execute()

    def captureCurrentFrame(self):
        buffer = self.data_stream.WaitForFinishedBuffer(5000)

        ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
        converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_BGRa8)

        # Queue buffer so that it can be used again
        self.data_stream.QueueBuffer(buffer)

        # Get raw image data from converted image and construct a QImage from it
        self.m_current_frame = converted_ipl_image.get_numpy_1D()
        self.m_current_frame = cv2.cvtColor(numpy.reshape(self.m_current_frame, [converted_ipl_image.Height(), converted_ipl_image.Width(), -1]), cv2.COLOR_BGR2GRAY)
        self.m_current_frame = cv2.resize(self.m_current_frame, (900, 600), cv2.INTER_AREA)
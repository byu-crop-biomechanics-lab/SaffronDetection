#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import asyncio


    

# class Oak:
#     def __init__(self):
def createPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    
    xoutRgb.setStreamName("video")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)

    # Create output
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(12)
    
    camRgb.video.link(xoutRgb.input)

    return pipeline

class Oak_system:
    
    def __init__(self):
        
        with contextlib.ExitStack() as stack:
            self.deviceInfos = dai.Device.getAllAvailableDevices()
            # usbSpeed = dai.UsbSpeed.SUPER
            openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4
            
            self.streams = []
            self.devices = []
            
        for deviceInfo in self.deviceInfos:
            deviceInfo: dai.DeviceInfo
            device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo))
            self.devices.append(device)
            print("===Connected to ", deviceInfo.getMxId())
            # mxId = device.getMxId()
            # cameras = device.getConnectedCameras()
            # usbSpeed = device.getUsbSpeed()
            # eepromData = device.readCalibration2().getEepromData()
            # print("   >>> MXID:", mxId)
            # print("   >>> Num of cameras:", len(cameras))
            # print("   >>> USB speed:", usbSpeed)
            # if eepromData.boardName != "":
            #     print("   >>> Board name:", eepromData.boardName)
            # if eepromData.productName != "":
            #     print("   >>> Product name:", eepromData.productName)

            pipeline = createPipeline()
            device.startPipeline(pipeline)
            self.streams.append( device.getOutputQueue(name = "video", maxSize = 1, blocking = False) )

            # # Output queue will be used to get the rgb frames from the output defined above
            # q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            # stream_name = "rgb-" + mxId + "-" + eepromData.productName
            # self.qRgbMap.append((q_rgb, stream_name))
            
            # for stream in self.streams:
            #     if stream.has():
            #         stream.get()
            
    # def iter(self):
        
        
                
class Oak:
    def __init__(self, ip):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutVideo = pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("rgb")
        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # camRgb.setVideoSize(1920, 1080)
        # camRgb.setFps(30)

        # Linking
        camRgb.preview.link(xoutVideo.input)
        
        device_info = dai.DeviceInfo(ip)
        
        # device_info.state = dai.X_LINK_BOOTED
        
        with dai.Device(pipeline, device_info) as device:
            self.video = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            
            # while True:
            #     self.frame = (self.video.get()).getCvFrame()
            
    def iter(self):

        videoIn = self.video.get()
        self.frame = videoIn.getCvFrame()
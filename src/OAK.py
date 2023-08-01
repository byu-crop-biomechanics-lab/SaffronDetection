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

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)
    # camRgb.setInterleaved(False)

    # Create output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("video")
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(1)
    
    camRgb.preview.link(xoutRgb.input)

    return pipeline
        
        
    # def config_cam(self):

    #     # Properties
    #     self.camRgb.setPreviewSize(300, 300)
    #     self.camRgb.setInterleaved(False)
    #     self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    #     # Linking
    #     self.camRgb.preview.link(self.xoutRgb.input)
    # def iter(self):
    #     # Connect to device and start pipeline
    #     with dai.Device(self.pipeline) as device:

    #         print('Connected cameras:', device.getConnectedCameraFeatures())
    #         # Print out usb speed
    #         print('Usb speed:', device.getUsbSpeed().name)
    #         # Bootloader version
    #         if device.getBootloaderVersion() is not None:
    #             print('Bootloader version:', device.getBootloaderVersion())
    #         # Device name
    #         print('Device name:', device.getDeviceName())

    #         # Output queue will be used to get the rgb frames from the output defined above
    #         qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    #         while True:
    #             inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

    #             # Retrieve 'bgr' (opencv format) frame
    #             cv2.imshow("rgb", inRgb.getCvFrame())

    #             if cv2.waitKey(1) == ord('q'):
    #                 break

class Oak_system:
    
    def __init__(self):
        
        with contextlib.ExitStack() as stack:
            self.deviceInfos = dai.Device.getAllAvailableDevices()
            usbSpeed = dai.UsbSpeed.SUPER
            openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4
            
            self.streams = []
            self.devices = []
            
        for deviceInfo in self.deviceInfos:
            deviceInfo: dai.DeviceInfo
            device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
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

            
        
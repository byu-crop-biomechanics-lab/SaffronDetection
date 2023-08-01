#!/usr/bin/env python3

import cv2
import depthai as dai

class Oak:
    def __init__(self):
        
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define source and output
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")
        
    def config_cam(self):

        # Properties
        self.camRgb.setPreviewSize(300, 300)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        self.camRgb.preview.link(self.xoutRgb.input)
    def iter(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            print('Connected cameras:', device.getConnectedCameraFeatures())
            # Print out usb speed
            print('Usb speed:', device.getUsbSpeed().name)
            # Bootloader version
            if device.getBootloaderVersion() is not None:
                print('Bootloader version:', device.getBootloaderVersion())
            # Device name
            print('Device name:', device.getDeviceName())

            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            while True:
                inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

                # Retrieve 'bgr' (opencv format) frame
                cv2.imshow("rgb", inRgb.getCvFrame())

                if cv2.waitKey(1) == ord('q'):
                    break

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
            
    def frame(self, device):
        stream = device.getOutputQueue(name = "video", maxSize = 1, blocking = False) 
        if stream.has():
            frame = stream.get()
        else:
            frame = None
        return frame
                
                
class Oak:
    def __init__(self, ip):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutVideo = pipeline.create(dai.node.XLinkOut)

        # Xout Properties
        xoutVideo.setStreamName("rgb")
        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(30)

        # Cam Rgb Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setVideoSize(1920, 1080)
        # camRgb.setFps(30)

        # Linking
        camRgb.video.link(xoutVideo.input)
        
        device_info = dai.DeviceInfo(ip)
        
        # device_info.state = dai.X_LINK_BOOTED
        
        with dai.Device(pipeline, device_info) as device:
            
            self.video = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            
            # while True:
            #     self.frame = (self.video.get()).getCvFrame()
            
    def iter(self):
        if self.video.has():
            self.frame = self.video.tryGet()
        
        
        
        
    
    #         #--------Code added here--------#
            
            
    #         # get image and show
    #         for view_name in ["data", "rgb", "disparity", "left", "right"]:
    #             # Skip if view_name was not included in frame
    #             try:
    #                 # Decode the image and render it in the correct kivy texture
                    
    #                 # Data was added by me to show us debugging and useful information
    #                 if view_name == "data":
                        
    #                     img = self.image_decoder.decode(
    #                         getattr(frame, 'rgb').image_data
    #                     )
    #                     img[:][:][:] = 0
                        
    #                     imu_packet = getattr(frame, 'imu_packets').packets[0]
                        
    #                     imu_x = imu_packet.gyro_packet.gyro.x
    #                     imu_y = imu_packet.gyro_packet.gyro.y
    #                     imu_z = imu_packet.gyro_packet.gyro.z
                                                
    #                     acc_x = imu_packet.accelero_packet.accelero.x
    #                     acc_y = imu_packet.accelero_packet.accelero.y
    #                     acc_z = imu_packet.accelero_packet.accelero.z
                        
    #                     cv2.putText(img, "IMU packet things:",(30,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    #                     cv2.putText(img, 'G X: %.4s' % str(imu_x),(350,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'G Y: %.4s' % str(imu_y),(350,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'G Z: %.4s' % str(imu_z),(350,350),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
    #                     cv2.putText(img, 'A X: %.4s' % str(acc_x),(30,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'A Y: %.4s' % str(acc_y),(30,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'A Z: %.4s' % str(acc_z),(30,350),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
    #                     cv2.putText(img, 'Gantry things',(600,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'X: %.4s' %str(self.gantry_x),(600,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    #                     cv2.putText(img, 'Y: %.4s' %str(self.gantry_y),(600,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
                        
    #                     meta_data = getattr(frame, 'rgb').meta.category
    #                     cv2.putText(img, 'meta: %.4s' %str(meta_data),(30,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)                        
                        
                        
    #                 # color filtering based on hue
    #                 elif view_name == 'rgb':
    #                     img = self.image_decoder.decode(
    #                         getattr(frame, view_name).image_data
    #                     )                                                
    #                     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #                     purple_lower = np.array([120,70,50])
    #                     purple_upper = np.array([135,255,255])
    #                     purple_amount = 400
    #                     purple_full_mask = cv2.inRange(img, purple_lower, purple_upper)
    #                     rgb_size = (img.shape[1],img.shape[0])                        
                        
    #                     # calculate center of purple object
    #                     cX = None
    #                     cY = None
    #                     if np.count_nonzero(purple_full_mask) >= purple_amount:
    #                         ret,thresh = cv2.threshold(purple_full_mask,127,255,0)
        
    #                         M = cv2.moments(thresh)
                            
    #                         cX = int(M["m10"] / M["m00"])
    #                         cY = int(M["m01"] / M["m00"])
                        
    #                     img = cv2.bitwise_and(img, img, mask=purple_full_mask)
    #                     img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR) 
                        
    #                     # put text and highlight the center
    #                     if cX and cY:
    #                         cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    #                         # text = "centroid: " + str(cX) + " " + str(cY)
    #                         # cv2.putText(img, text, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
    #                     disparity_img = self.image_decoder.decode(
    #                         getattr(frame, "disparity").image_data
    #                     )
    #                     disparity_img = cv2.resize(disparity_img,(img.shape[1], img.shape[0]))
                        
    #                     '''
    #                     # this is to show what the disparity image shows us at the location of the pom pom
    #                     if cX and cY:

    #                         cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    #                         text = "Center: " + str(disparity_img[cX][cY])
    #                         cv2.putText(img, text, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #                     '''

    #                 else:
    #                     img = self.image_decoder.decode(
    #                         getattr(frame, view_name).image_data
    #                     )


    #                 #----------end of my custom code----------#
                    
                    
    #                 texture = Texture.create(
    #                     size=(img.shape[1], img.shape[0]), icolorfmt="bgr"
    #                 )
    #                 texture.flip_vertical()
    #                 texture.blit_buffer(
    #                     img.tobytes(),
    #                     colorfmt="bgr",
    #                     bufferfmt="ubyte",
    #                     mipmap_generation=False,
    #                 )
                    
    #                 self.root.ids[view_name].texture = texture

    #             except Exception as e:
    #                 print(e)

'''
To do:
    Now that we can send to the microcontroller, send gantry commands from amiga to gantry
    Await for a response from gantry to be send before new command is sent from the brain
    have a home button that when pressed the gantry homes only once
''' 


import argparse
import asyncio
import os
from typing import List
from typing import Optional

import grpc


# canbus things
from farm_ng.canbus import canbus_pb2
from farm_ng.canbus.canbus_client import CanbusClient
from farm_ng.canbus.packet import AmigaControlState
from farm_ng.canbus.packet import AmigaTpdo1
from farm_ng.canbus.packet import make_amiga_rpdo1_proto
from farm_ng.canbus.packet import parse_amiga_tpdo1_proto

# camera things
# from farm_ng.oak import oak_pb2
# from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
import turbojpeg
# from OAK import Oak
from OAK import Oak_system
from OAK import Oak

import depthai as dai

# things I've added #
from gantry import GantryControlState
from gantry import GantryRpdo1
from gantry import GantryTpdo1
from gantry import make_gantry_tpdo1_proto
from gantry import parse_gantry_rpdo1_proto

import cv2
import numpy as np
import contextlib

#----#

os.environ["KIVY_NO_ARGS"] = "1"


from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402


class CameraColorApp(App):
    def __init__(self, address: str, canbus_port: int, stream_every_n: int) -> None:
        super().__init__()
        self.address: str = address
        # self.camera_port : int = camera_port
        self.canbus_port: int = canbus_port
        self.stream_every_n = stream_every_n
        
        self.amiga_rpdo1: AmigaTpdo1 = AmigaTpdo1()
        self.amiga_state = AmigaControlState.STATE_AUTO_READY
        self.amiga_rate = 0
        self.amiga_speed = 0
        
        self.gantry_tpdo1: GantryTpdo1 = GantryTpdo1()
        self.gantry_rpdo1: GantryRpdo1 = GantryRpdo1()
        self.gantry_state = GantryControlState.STATE_AUTO_READY
        self.gantry_x = 0
        self.gantry_y = 0
        self.gantry_feed = 1000
        self.gantry_jog = 1
        self.sender = 0
        self.receiver = 0

        self.image_decoder = turbojpeg.TurboJPEG()
        
        self.tasks: List[asyncio.Task] = []

    def build(self):
        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()
        
    # def on_home_btn(self) -> None:
    #     # home the gantry
    #     # maybe a variable to say home, Go1, jog, or alarm states
    #     pass

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        # # configure the camera client
        # camera_config: ClientConfig = ClientConfig(
        #     address=self.address, port=self.camera_port
        # )
        # camera_client: OakCameraClient = OakCameraClient(camera_config)
        # self.oaks = Oak_system()
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
            pipeline = self.createPipeline()
            device.startPipeline(pipeline)
            self.streams.append( device.getOutputQueue(name = "video", maxSize = 12, blocking = False) )

        # self.oaks = [Oak("10.95.76.10"), Oak("10.95.76.11")]
        # self.oak = Oak("10.95.76.11")
        # self.oak_1 = self.oaks.devices[0]
        # self.oak_2 = self.oaks.devices[1]

        # configure the canbus client
        canbus_config: ClientConfig = ClientConfig(
            address=self.address, port=self.canbus_port
        )
        canbus_client: CanbusClient = CanbusClient(canbus_config)

        # # Camera task(s)
        # self.tasks.append(
        #     asyncio.ensure_future(self.stream_camera(camera_client))
        # )
        self.tasks.append(
            asyncio.ensure_future(self.stream_Oak())
        )

        # Canbus task(s)
        self.tasks.append(
            asyncio.ensure_future(self.stream_canbus(canbus_client))
        )
        self.tasks.append(
            asyncio.ensure_future(self.send_can_msgs(canbus_client))
        )


        return await asyncio.gather(run_wrapper(), *self.tasks)


    async def stream_canbus(self, client: CanbusClient) -> None:
        """This task:

        - listens to the canbus client's stream
        - filters for AmigaTpdo1 messages
        - extracts useful values from AmigaTpdo1 messages
        """
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None

        while True:
            # check the state of the service
            state = await client.get_state()

            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None

                print("Canbus service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            if (
                response_stream is None
                and state.value != service_pb2.ServiceState.UNAVAILABLE
            ):
                # get the streaming object
                response_stream = client.stream_raw()

            try:
                # try/except so app doesn't crash on killed service
                response: canbus_pb2.StreamCanbusReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                print(e)
                response_stream.cancel()
                response_stream = None
                continue

            for proto in response.messages.messages:                    
                # Check if message is for the gantry
                gantry_rpdo1: Optional[GantryRpdo1] = parse_gantry_rpdo1_proto(proto)
                if gantry_rpdo1:
                    # Store the value for possible other uses
                    self.gantry_rpdo1 = gantry_rpdo1
                    # print("Received some RPDO1")
    
    #---------depthAI stuff goes here--------#         
    # async def stream_camera(self, client: OakCameraClient) -> None:
    #     """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
    #     from the oak camera."""
    #     while self.root is None:
    #         await asyncio.sleep(0.01)

    #     response_stream = None

    #     while True:
    #         # check the state of the service
    #         state = await client.get_state()

    #         if state.value not in [
    #             service_pb2.ServiceState.IDLE,
    #             service_pb2.ServiceState.RUNNING,
    #         ]:
    #             # Cancel existing stream, if it exists
    #             if response_stream is not None:
    #                 response_stream.cancel()
    #                 response_stream = None
    #             print("Camera service is not streaming or ready to stream")
    #             await asyncio.sleep(0.1)
    #             continue

    #         # Create the stream
    #         if response_stream is None:
    #             response_stream = client.stream_frames(every_n=self.stream_every_n)

    #         try:
    #             # try/except so app doesn't crash on killed service
    #             response: oak_pb2.StreamFramesReply = await response_stream.read()
    #             assert response and response != grpc.aio.EOF, "End of stream"
    #         except Exception as e:
    #             print(e)
    #             response_stream.cancel()
    #             response_stream = None
    #             continue

    #         # get the sync frame
    #         frame: oak_pb2.OakSyncFrame = response.frame


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
                      
    def createPipeline(self):
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
        xoutRgb.input.setQueueSize(30)
        
        camRgb.video.link(xoutRgb.input)

        return pipeline

    async def stream_Oak(self):
        
        while self.root is None:
            await asyncio.sleep(0.01)
        # streams = []
        #-------RGBs-------#
        # for index, device in enumerate(self.oaks.devices):
        #     stream = device.getOutputQueue(name = "video", maxSize = 12, blocking = False)

            # self.oak.iter()
        
        # rgb_imgs = []
        # for index, oak in enumerate(self.oaks):
        
            
        # if self.oak.video.has():
        #     rgb_img = self.oak.video.get().getCvFrame()
            
        # else:
        #     rgb_img = 20 * np.ones(shape=[800, 1000, 3], dtype=np.uint8)
        for index, stream in enumerate(self.streams):
            if stream.has():
                rgb_img = (stream.get()).getCvFrame()
                
                texture = Texture.create(
                    size=(rgb_img.shape[1], rgb_img.shape[0]), icolorfmt="bgr"
                )
            
                texture.flip_vertical()
                texture.blit_buffer(
                    rgb_img.tobytes(),
                    colorfmt="bgr",
                    bufferfmt="ubyte",
                    mipmap_generation=False,
                )
                
            
                # index = 0
                self.root.ids[("rgb_" + str(index + 1))].texture = texture
        
        
        #-------depths-------#
        
        #-------Data-------#
        data_img = 20 * np.ones(shape=[800, 1000, 3], dtype=np.uint8)
        # cv2.putText(data_img, str(self.oaks.streams[0].has()), (30,150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        texture = Texture.create(
            size=(data_img.shape[1], data_img.shape[0]), icolorfmt="bgr"
        )
        
        texture.flip_vertical()
        texture.blit_buffer(
            data_img.tobytes(),
            colorfmt="bgr",
            bufferfmt="ubyte",
            mipmap_generation=False,
        )
        
        self.root.ids["data"].texture = texture
        
        # img = self.oaks.devices[0].q_rgb.get().getCvFrame()
        
                    
                    
    async def send_can_msgs(self, client: CanbusClient) -> None:
        """This task ensures the canbus client sendCanbusMessage method has the pose_generator it will use to send
        messages on the CAN bus to control the Amiga robot."""
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None
        while True:
            # check the state of the service
            state = await client.get_state()

            # Wait for a running CAN bus service
            if state.value != service_pb2.ServiceState.RUNNING:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                print("Waiting for running canbus service...")
                await asyncio.sleep(0.1)
                continue

            if response_stream is None:
                print("Start sending CAN messages")
                # self.sender = self.sender + 1
                response_stream = client.stub.sendCanbusMessage(self.pose_generator())

            '''
            # This isn't working
            try:
                async for response in response_stream:
                    # Sit in this loop and wait until canbus service reports back it is not sending
                    assert response.success
            except Exception as e:
                print(e)
                response_stream.cancel()
                response_stream = None
                continue
            '''
            

            await asyncio.sleep(0.1)

# this is where you will determine whether or not to move the gantry based on the purple color sent.
    async def pose_generator(self, period: float = 0.02):

        while self.root is None:
            await asyncio.sleep(0.01)
        # put the x and y coordinate and feed stuff right here
        while True:
            self.gantry_x = self.gantry_x + 1
            msg: canbus_pb2.RawCanbusMessage = make_gantry_tpdo1_proto(
                # state_req = GantryControlState.STATE_AUTO_ACTIVE,
                # cmd_feed = self.gantry_feed,
                T_x = self.gantry_x,
                T_y = self.gantry_y,
            )
            # print("Sent TPDO")
            yield canbus_pb2.SendCanbusMessageRequest(message=msg)
            await asyncio.sleep(period)
            
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="color-detector-oak")
    parser.add_argument(
        "--address", type=str, default="localhost", help="The server address"
    )
    # parser.add_argument(
    #     "--camera-port",
    #     type=int,
    #     required=True,
    #     help="The grpc port where the camera service is running.",
    # )
    parser.add_argument(
        "--canbus-port",
        type=int,
        required=True,
        help="The grpc port where the canbus service is running.",
    )    
    # parser.add_argument(
    #     "--address", type=str, default="localhost", help="The camera address"
    # )
    parser.add_argument(
        "--stream-every-n", 
        type=int, 
        default=1, 
        help="Streaming frequency"
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            CameraColorApp(args.address, args.canbus_port, args.stream_every_n).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()
    
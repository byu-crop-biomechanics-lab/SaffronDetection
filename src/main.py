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

# camera things
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
        
        self.gantry_tpdo1: GantryTpdo1 = GantryTpdo1()
        self.gantry_rpdo1: GantryRpdo1 = GantryRpdo1()
        self.gantry_state = GantryControlState.STATE_AUTO_READY
        self.gantry_x = 0
        self.gantry_y = 0
        self.gantry_feed = 1000
        self.gantry_jog = 1

        # self.image_decoder = turbojpeg.TurboJPEG()
        
        self.tasks: List[asyncio.Task] = []

    def build(self):
        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        # # configure the camera
        # self.oaks = [Oak("10.95.76.10"), Oak("10.95.76.11")]
        self.oak = Oak("10.95.76.11")
        # self.oak_1 = self.oaks.devices[0]
        # self.oak_2 = self.oaks.devices[1]

        # configure the canbus client
        canbus_config: ClientConfig = ClientConfig(
            address=self.address, port=self.canbus_port
        )
        canbus_client: CanbusClient = CanbusClient(canbus_config)

        # Camera task(s)
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
        - filters for Tpdo1 messages
        - extracts useful values from Tpdo1 messages
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

    async def stream_Oak(self):
        
        while self.root is None:
            await asyncio.sleep(0.01)

        # self.oak.iter()
        
        #-------RGBs-------#
        
        # rgb_img = self.oak.frame
        # texture = Texture.create(
        #     size=(rgb_img.shape[1], rgb_img.shape[0]), icolorfmt="bgr"
        # )
    
        # texture.flip_vertical()
        # texture.blit_buffer(
        #     rgb_img.tobytes(),
        #     colorfmt="bgr",
        #     bufferfmt="ubyte",
        #     mipmap_generation=False,
        # )
        
    
        # index = 0
        # self.root.ids[("rgb_" + str(index + 1))].texture = texture
        
        
        #-------depths-------#
        
        #-------Data-------#
        data_img = 20 * np.ones(shape=[800, 1000, 3], dtype=np.uint8)
        cv2.putText(data_img, str(self.oak.video.has()), (30,150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


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
    parser.add_argument(
        "--canbus-port",
        type=int,
        required=True,
        help="The grpc port where the canbus service is running.",
    )    
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
    
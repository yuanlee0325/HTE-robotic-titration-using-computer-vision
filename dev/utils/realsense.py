###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime


def start_realsense(fname='polymer_2_spectral_measurements_',
                    folder=r'C:\Users\bdutta\work\pys\Cam\Unilever_Test_120123',
                    frame_interval=60,# in seconds
                    stop=2,#stop hours
                    take_image=False,
                    sensitivity = 10):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = True
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # Set the exposure anytime duri~ng the operation
    sensor.set_option(rs.option.exposure, sensitivity)
    try:
        count=0
        old_t=0
        count_dummy=0
        frame_interval=int(100/3.33*frame_interval)
        stop = int(stop * (60 * 60) * int(100/3.33))# stop(in hours) *(3600 s)
        #frame_interval=2000
        while True: 
            dt=datetime.today()
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                #images = np.hstack((color_image, depth_colormap))
                images = color_image
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)
            if count % frame_interval==0:
                new_t=time.time()
                # save image as defined type, png or jpg
                path=os.path.join(folder,fname+str(count_dummy)+'.jpg')
                if take_image : cv2.imwrite(path,images)
                print('time elapsed: ',new_t-old_t)
                old_t=new_t   
                count_dummy+=1
            if key & 0xFF == ord('q') or key == 27 or count==stop:
                cv2.destroyAllWindows()
                break  
            count+=1
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("running!")
    os.chdir(r"C:\Users\scrc112\Desktop\work\QJ")
    start_realsense()
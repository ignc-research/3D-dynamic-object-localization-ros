#!/usr/bin/env python3
"""
 @Author: Hichem Dhouib
 @Date: 2021
 @Last Modified by:   Hichem Dhouib 
 @Last Modified time:
"""

import sys
import cv2
import numpy as np
import os
from cv_bridge import CvBridge
from mmdet.apis import inference_detector, init_detector
import rospy
import message_filters
from sensor_msgs.msg import Image , CompressedImage , CameraInfo
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from logging import debug
from time import time
from contextlib import contextmanager
from funcy import print_durations
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

# Setup check for docker
import mmdet
import mmcv
import torch
print("opencv version: ", cv2.__version__)
print("numpy version: ", np.__version__)
print("torch version: ",torch.__version__, "| torch cuda available: ",torch.cuda.is_available())
print("mmdetection version: ",mmdet.__version__)
print("mmcv version: ", mmcv.__version__)
print("compiling cuda version: ", get_compiling_cuda_version())
print("compiler version: ", get_compiler_version())
print("python3: ",sys.version)

DELETEALL_MARKER_ID = 20
SCALE = 1
CONFIG_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/scripts/yolov3_d53_320_273e_coco.py'
MODEL_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/scripts/latest.pth'

det2dobj = Detection2D()

@contextmanager
def timer(descrption: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    rospy.logdebug(f"{descrption}: {ellapsed_time}")

def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    X = (pixel_x - camera_intrinsics[2])/camera_intrinsics[0] *depth
    Y = (pixel_y - camera_intrinsics[5])/camera_intrinsics[4] *depth
    return -X, -Y, depth

def delete_markers():
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.action = Marker.DELETEALL
    marker.id = DELETEALL_MARKER_ID
    return marker

def extract_detection_results(subResult, det2dobj):
    det2dobj.bbox.center.x = (subResult[0] + subResult[2]) / 2
    det2dobj.bbox.center.y = (subResult[1] + subResult[3]) / 2
    det2dobj.bbox.size_x = subResult[2] - subResult[0]
    det2dobj.bbox.size_y = subResult[3] - subResult[1]
    objHypothesis = ObjectHypothesisWithPose()
    objHypothesis.score = subResult[4]
    det2dobj.results.append(objHypothesis)
    return det2dobj, det2dobj.objHypothesis.score

def make_marker(class_counter, robot_counter, marker_location, colors):
    marker = Marker()
    marker.header.frame_id =  "base_link"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale.x = SCALE / 2
    marker.scale.y = SCALE / 2
    marker.scale.z = SCALE
    marker.header.stamp  = rospy.get_rostime()
    marker.id = class_counter + robot_counter
    marker.pose.position.y, marker.pose.position.z, marker.pose.position.x = marker_location
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = colors
    return marker

class Detector:
    def __init__(self, model):

        self.bridge = CvBridge()
        self.pub_topic_color = "/mmdet/pose_estimation/det2d/compressed"
        self.image_pub = rospy.Publisher(self.pub_topic_color, CompressedImage, queue_size=3)

        self.marker_array_msg = MarkerArray()
        self.pub_topic_marker_array = "/mmdet/visualization_marker_array" 
        self.marker_array_pub = rospy.Publisher(self.pub_topic_marker_array, MarkerArray, queue_size=3)

        self.sub_topic_color = "/zed2/zed_node/rgb/image_rect_color"
        self.sub_topic_depth = "/zed2/zed_node/depth/depth_registered"
        self.sub_topic_cameraInfo =  "/zed2/zed_node/depth/camera_info"
        self.image_sub = message_filters.Subscriber(self.sub_topic_color,Image)
        self.depth_sub = message_filters.Subscriber(self.sub_topic_depth,Image)
        self.camera_intrinsics_sub = message_filters.Subscriber(self.sub_topic_cameraInfo, CameraInfo)

        self.model = model
        self.score_thr= 0.6
        self.marker_location = None
        self.scale = 1
        self.depth_value = 0

        self.visualization_3d = rospy.get_param("visualization_3d")
        self.visualization_2d = rospy.get_param("visualization_2d")

        self.camera_intrinsics = [266.2339172363281, 0.0, 335.1106872558594, 0.0, 266.2339172363281, 176.05209350585938, 0.0, 0.0, 1.0]

        self.robot_dict = { 0: "tiago", 1: "pepper"  , 2: "kuka" }

        self.bbox3D_tiago_colors = [1, 0 , 0 , 0.6]
        self.bbox3D_pepper_colors = [0 , 1 , 0, 0.6]
        self.bbox3D_kuka_colors = [0 , 0 , 1 , 0.6]
        self.colors_dict_3d = {0 : self.bbox3D_tiago_colors , 1 : self.bbox3D_pepper_colors , 2 : self.bbox3D_kuka_colors }

        self.bbox2D_tiago_colors = (255, 0 , 0 )
        self.bbox2D_pepper_colors = (0 , 255 , 0)
        self.bbox2D_kuka_colors = (0 , 0 , 255 )
        self.colors_dict_2d = {0 : self.bbox2D_tiago_colors , 1 : self.bbox2D_pepper_colors , 2 : self.bbox2D_kuka_colors }

    @print_durations()
    def callback(self, image, depth_data):

        deleteMarker = delete_markers()
        self.marker_array_msg.markers.append(deleteMarker)
        image_np = np.frombuffer(image.data, dtype = np.uint8).reshape(image.height, image.width, -1)
        image_rgba = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
        detectionResults = inference_detector(self.model,  image_rgba[ : , : , :3])
        dImage = np.frombuffer(depth_data.data,  dtype = np.float32).reshape(depth_data.height, depth_data.width, -1)
        det2dobj.header = image.header
        det2dobj.source_img = image

        for counter, detectedRobots in enumerate(detectionResults):
            for sub_counter, subResult in enumerate(detectedRobots):
                if subResult.shape != (0, 5):
                    det_2d_result, score = extract_detection_results(subResult, det2dobj)
                    #rospy.logdebug("score for %s  | nr: %s | score: %s", self.robot_dict[counter] ,sub_counter, score)

                    if score > self.score_thr:
                        if self.visualization_2d is True:
                            start_point = (int(det_2d_result.bbox.center.x - det_2d_result.bbox.size_x/2) ,int(det_2d_result.bbox.center.y-det_2d_result.bbox.size_y/2))
                            end_point = (int(det_2d_result.bbox.center.x + det_2d_result.bbox.size_x/2) , int(det_2d_result.bbox.center.y+det_2d_result.bbox.size_y/2))
                            cv_img = cv2.rectangle(image_np, start_point, end_point, self.colors_dict_2d[counter], 3)
                            #rospy.logdebug("2D bbox for %s | nr: %s | score: %s", self.robot_dict[counter] ,sub_counter, score)
                            cv_img = self.bridge.cv2_to_compressed_imgmsg(cv_img)
                            self.image_pub.publish(cv_img)

                        if self.visualization_3d is True:
                            self.depth_value = dImage[int(det_2d_result.bbox.center.y), int(det_2d_result.bbox.center.x)]
                            #rospy.logdebug("3D bbox for %s | nr: %s score: %s | depth: %s ", self.robot_dict[counter] ,sub_counter, score, self.depth_value)
                            marker_location = convert_depth_pixel_to_metric_coordinate(self.depth_value, det_2d_result.bbox.center.x, det_2d_result.bbox.center.y, self.camera_intrinsics)
                            marker = make_marker(counter, sub_counter, marker_location, self.colors_dict_3d[counter])
                            #rospy.logdebug(" -- Appending new marker to markerarray -- %s %s %s", marker.pose.position.y , marker.pose.position.z , marker.pose.position.x )
                            self.marker_array_msg.markers.append(marker)

        self.marker_array_pub.publish(self.marker_array_msg.markers)

def main():

    rospy.init_node('mmdetector', log_level=rospy.DEBUG)
    model = init_detector(CONFIG_PATH, MODEL_PATH, device='cuda:0')
    detector = Detector(model)
    ts = message_filters.ApproximateTimeSynchronizer([detector.image_sub, detector.depth_sub], queue_size=3, slop=0.5, allow_headerless=True)
    ts.registerCallback(detector.callback)
    rospy.spin()

if __name__=='__main__':
    main()



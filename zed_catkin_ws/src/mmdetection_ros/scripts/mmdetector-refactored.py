#!/usr/bin/env python3
"""
 @Author: Hichem Dhouib
 @Date: 2022
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

CONFIG_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/scripts/yolov3_d53_320_273e_coco.py'
MODEL_PATH = '/workspace/zed_catkin_ws/src/mmdetection_ros/scripts/latest.pth'
DETECTION_SCORE_THRESHOLD = 0.6
SCALE = 1
DELETEALL_MARKER_ID = 20

CAMERA_INTRINSICS = [266.2339172363281, 0.0, 335.1106872558594, 0.0, 266.2339172363281, 176.05209350585938, 0.0, 0.0, 1.0] # cam intrinsics for VGA
#[1066.1673583984375, 0.0, 976.6065673828125, 0.0, 1066.1673583984375, 546.295654296875, 0.0, 0.0, 1.0] # cam intrinsics for 1080HD
ROBOTS = { 0: "tiago", 1: "pepper"  , 2: "kuka" }
BBOX_3D_TIAGO_COLORS = [1, 0 , 0 , 0.6]
BBOX_3D_PEPPER_COLORS = [0 , 1 , 0, 0.6]
BBOX_3D_KUKA_COLORS = [0 , 0 , 1 , 0.6]
COLORS_3D_DICT = {0 : BBOX_3D_TIAGO_COLORS, 1 : BBOX_3D_PEPPER_COLORS, 2 : BBOX_3D_KUKA_COLORS}

BBOX_2D_TIAGO_COLORS = (255, 0 , 0 )
BBOX_2D_PEPPER_COLORS = (0 , 255 , 0)
BBOX_2D_KUKA_COLORS = (0 , 0 , 255 )
COLORS_2D_DICT = {0 : BBOX_2D_TIAGO_COLORS, 1 : BBOX_2D_TIAGO_COLORS, 2 : BBOX_2D_KUKA_COLORS}

@contextmanager
def timer(descrption: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    rospy.logdebug(f"{descrption}: {ellapsed_time}")

class Detector:
    def __init__(self, model):

        self.bridge = CvBridge()
        self.pub_topic_color = "/mmdet/pose_estimation/det2d/compressed"
        self.image_pub = rospy.Publisher(self.pub_topic_color, CompressedImage, queue_size=1)

        self.pub_topic_marker_array = "/mmdet/visualization_marker_array"
        self.marker_array_pub = rospy.Publisher(self.pub_topic_marker_array, MarkerArray, queue_size=1)

        self.sub_topic_color = "/zed2/zed_node/rgb/image_rect_color"
        self.sub_topic_depth = "/zed2/zed_node/depth/depth_registered"
        self.sub_topic_cameraInfo =  "/zed2/zed_node/depth/camera_info"
        self.image_sub = message_filters.Subscriber(self.sub_topic_color,Image)
        self.depth_sub = message_filters.Subscriber(self.sub_topic_depth,Image)
        self.camera_intrinsics_sub = message_filters.Subscriber(self.sub_topic_cameraInfo, CameraInfo)

        self.model = model
        self.visualization_3d = rospy.get_param("visualization_3d")
        self.visualization_2d = rospy.get_param("visualization_2d")


    def convert_depth_pixel_to_metric_coordinate(self, depth, pixel_x, pixel_y):
        X = (pixel_x - CAMERA_INTRINSICS[2])/CAMERA_INTRINSICS[0] *depth
        Y = (pixel_y - CAMERA_INTRINSICS[5])/CAMERA_INTRINSICS[4] *depth
        return -X, -Y, depth

    def extract_detection_results(self, subResult, det2dobj):
        det2dobj.bbox.center.x = (subResult[0] + subResult[2]) / 2
        det2dobj.bbox.center.y = (subResult[1] + subResult[3]) / 2
        det2dobj.bbox.size_x = subResult[2] - subResult[0]
        det2dobj.bbox.size_y = subResult[3] - subResult[1]
        objHypothesis = ObjectHypothesisWithPose()
        objHypothesis.score = subResult[4]
        det2dobj.results.append(objHypothesis)
        return det2dobj, objHypothesis.score

    def delete_markers(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.action = Marker.DELETEALL
        marker.id = DELETEALL_MARKER_ID
        return marker

    def make_marker(self, det_cls_counter, det_inst_counter, marker_location):
        marker = Marker()
        marker.header.frame_id =  "base_link"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = SCALE / 2
        marker.scale.y = SCALE / 2
        marker.scale.z = SCALE
        marker.header.stamp  = rospy.get_rostime()
        marker.id = det_cls_counter + det_inst_counter
        marker.pose.position.y , marker.pose.position.z , marker.pose.position.x = marker_location
        marker.color.r , marker.color.g , marker.color.b , marker.color.a =  COLORS_3D_DICT[det_cls_counter]
        return marker

    #@print_durations()
    def callback(self, image, depth_data):

        delete_marker = self.delete_markers()
        marker_array_msg = MarkerArray()
        marker_array_msg.markers.append(delete_marker)

        det2dobj = Detection2D()
        det2dobj.header = image.header
        det2dobj.source_img = image

        image_np = np.frombuffer(image.data, dtype = np.uint8).reshape(image.height, image.width, -1)
        image_rgba = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
        detectionResults = inference_detector(self.model,  image_rgba[ : , : , :3])

        dImage =  np.frombuffer(depth_data.data,  dtype = np.float32).reshape(depth_data.height, depth_data.width, -1)
        
        compressed_cv_image = None
        for det_cls_counter, detectedRobots in enumerate(detectionResults):
            for det_instance_counter, subResult in enumerate(detectedRobots):
                if subResult.shape != (0, 5):
                    det_2d_result, score = self.extract_detection_results(subResult, det2dobj)
                    if score > DETECTION_SCORE_THRESHOLD:
                        if self.visualization_2d is True:
                            start_point = (int(det_2d_result.bbox.center.x - det_2d_result.bbox.size_x/2) ,int(det_2d_result.bbox.center.y - det_2d_result.bbox.size_y/2))
                            end_point = (int(det_2d_result.bbox.center.x + det_2d_result.bbox.size_x/2) , int(det_2d_result.bbox.center.y + det_2d_result.bbox.size_y/2))
                            cv_img = cv2.rectangle(image_np, start_point, end_point, COLORS_2D_DICT[det_cls_counter], 3)
                            compressed_cv_image = self.bridge.cv2_to_compressed_imgmsg(cv_img)
                        if self.visualization_3d is True:
                            depth_value = dImage[int(det_2d_result.bbox.center.y), int(det_2d_result.bbox.center.x)]
                            if depth_value[0] > 0.0:
                                marker_location = self.convert_depth_pixel_to_metric_coordinate(depth_value, det_2d_result.bbox.center.x, det_2d_result.bbox.center.y)
                                marker = self.make_marker(det_cls_counter, det_instance_counter, marker_location)
                                marker_array_msg.markers.append(marker)

        if self.visualization_2d is True and compressed_cv_image != None:
            self.image_pub.publish(compressed_cv_image)

        self.marker_array_pub.publish(marker_array_msg.markers)


def main():
    rospy.init_node('mmdetector', log_level=rospy.DEBUG)
    model = init_detector(CONFIG_PATH, MODEL_PATH, device='cuda:0')
    detector = Detector(model)
    ts = message_filters.ApproximateTimeSynchronizer([detector.image_sub, detector.depth_sub], queue_size=1, slop=0.5, allow_headerless=True)
    ts.registerCallback(detector.callback)
    rospy.spin()

if __name__=='__main__':
    main()



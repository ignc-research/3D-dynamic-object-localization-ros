#!/usr/bin/env python3
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import os
from datetime import datetime
import rospy
import cv2
import mmcv
#import matplotlib.pyplot as plt
import numpy as np
import message_filters
from mmdet.apis import init_detector, inference_detector
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from visualization_msgs.msg import Marker
#from sensor_msgs.msg import RegionOfInterest #http://docs.ros.org/en/jade/api/sensor_msgs/html/msg/RegionOfInterest.html
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose #http://docs.ros.org/en/kinetic/api/vision_msgs/html/msg/Detection2D.html
from cv_bridge import CvBridge, CvBridgeError


def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    X = (pixel_x - camera_intrinsics[2])/camera_intrinsics[0] *depth
    Y = (pixel_y - camera_intrinsics[5])/camera_intrinsics[4] *depth
    return X, Y, depth




class detection_node:
  def __init__(self):
    print("Starting ROS node...")
    self.bridge = CvBridge()
    self.sub_topic_color = "/camera/color/image_raw/compressed"
    self.sub_topic_depth = "/camera/depth/image_rect_raw"
    #self.sub_topic_camerainfo = "/camera/color/camera_info"
    self.pub_topic_color = "/pose_estimation/det2d/compressed"
    self.pub_topic_marker= "/pose_estimation/det2d/marker"
    self.pub_topic_det2d = "/pose_estimation/det2d"

    self.image_sub = message_filters.Subscriber(self.sub_topic_color,CompressedImage)
    self.depth_sub = message_filters.Subscriber(self.sub_topic_depth,Image)
    #self.info_sub = message_filters.Subscriber(self.sub_topic_camerainfo,CameraInfo)
    self.image_pub = rospy.Publisher(self.pub_topic_color, CompressedImage, queue_size=1)
    self.marker_pub = rospy.Publisher(self.pub_topic_marker, Marker, queue_size=10)
    self.det2d_pub = rospy.Publisher(self.pub_topic_det2d, Detection2D, queue_size=1)

    #config_file = './model/faster_rcnn_r50_fpn_1x_coco_pepper.py'
    #checkpoint_file = './model/epoch_12.pth'
    config_file = './model/yolov3_d53_mstrain-608_273e_coco_pepper.py'
    checkpoint_file = './model/epoch_60.pth'
    self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
    self.result_img = []
    self.score_thr=0.99
    # The flattened camera intrinsics matrix for the particular realsense depth camera utilized in the original project.
    # IMPORTANT: Needs to be replaced if used with different camera.
    self.camera_intrinsics = [903.9404907226562, 0.0, 632.892333984375, 0.0, 902.3751220703125, 374.73358154296875, 0.0, 0.0, 1.0]

    self.rate = 5 # ROS Node Hz
    print("Startup successfull")

    #self.id = 0
    #self.color_saved = False
    #self.depth_saved = False

  def callback(self, data, depth_data):#, camera_info):
    try:
        cv_img = self.bridge.compressed_imgmsg_to_cv2(data)#, data.encoding)#, desired_encoding="bgr8") # received image
        result = inference_detector(self.model, cv_img)

        depth_img = self.bridge.imgmsg_to_cv2(depth_data, depth_data.encoding)

        marker_location = None
        depth = 0

        if(len(result)>=1 and len(result[0])>=1):
            print("Detection occured")
            bboxes = result[0]
            bbox=bboxes[0] # select bbox with highest confidence

            det2d = Detection2D()
            det2d.header = data.header

            x1, y1, x2, y2, conf = bbox
            det2d.bbox.size_x = int(x2-x1)
            det2d.bbox.size_y = int(y2-y1)
            det2d.bbox.center.x = int((x2+x1)/2)
            det2d.bbox.center.y = int((y2+y1)/2)
            det2d.bbox.center.theta = 0

            result = ObjectHypothesisWithPose()
            result.id = 0
            result.score = conf
            det2d.results.append(result)

            print("Confidence of detection:")
            print(conf)

            if conf > self.score_thr:
                self.det2d_pub.publish(det2d)


                start_point = (det2d.bbox.center.x - int(det2d.bbox.size_x/2), det2d.bbox.center.y-int(det2d.bbox.size_y/2))

                end_point = (det2d.bbox.center.x + int(det2d.bbox.size_x/2), det2d.bbox.center.y+int(det2d.bbox.size_y/2))

                cv_img = cv2.rectangle(cv_img, start_point, end_point, (255, 0, 0), 2)
                cv_img = self.bridge.cv2_to_compressed_imgmsg(cv_img)
                self.image_pub.publish(cv_img)
                depth = depth_img[int(det2d.bbox.center.y/720 * 480) , int(det2d.bbox.center.x/1280 *848)]*0.001
                print("Detected Depth:")
                print(depth)
                marker_location = convert_depth_pixel_to_metric_coordinate(depth, det2d.bbox.center.x, det2d.bbox.center.y, self.camera_intrinsics)

        marker = Marker()
        marker.header.frame_id = "camera_depth_frame"
        marker.header.stamp    = rospy.get_rostime()
        marker.id = 1
        marker.type = marker.SPHERE # sphere
        marker.action = Marker.DELETEALL
        print(marker_location)

        if(marker_location != None):
            marker.action = Marker.ADD
            marker.pose.position.x = marker_location[2]
            marker.pose.position.y = -marker_location[0]
            marker.pose.position.z = marker_location[1]
            marker.color.a = 1
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
            scale = depth / 6
            if(scale > 1):
                scale = 1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

        self.marker_pub.publish(marker)

    except CvBridgeError as e:
      print(e)


def main(args):
  ic = detection_node()
  rospy.init_node('detection_node', anonymous=True)
  rospy.loginfo("Subscribing to the topic %s", ic.sub_topic_color)
  rate = rospy.Rate(ic.rate) # hz

  ts = message_filters.ApproximateTimeSynchronizer([ic.image_sub, ic.depth_sub], 10, 0.5, allow_headerless=True)#, ic.info_sub], 10, 0.5, allow_headerless=True)
  ts.registerCallback(ic.callback)
  rospy.spin()
  #try:
    #rospy.spin()
    #while not rospy.is_shutdown():
    #    rate.sleep()
  #except KeyboardInterrupt:
    #print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python

import rospy
import message_filters
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
import numpy as np

# global MarkerArray to store the markers of detected objects
markerArray = MarkerArray()


def delete_marker():
    """
    Initializes a DELETEALL Marker
    """
    marker = Marker()
    marker.header.frame_id = "/base_link"
    marker.id = 20000
    marker.action = Marker.DELETEALL
    return marker


def make_marker(x, y, z, scale):
    """
    Initialies a marker of Type Sphere.
        @param x: Real world X coordinate
        @param y: Real world Y coordinate
        @param z: Real world Z coordinate
        @param scale: scale of sphere

        @return: Marker
    """
    # Init Marker
    marker = Marker()
    marker.header.frame_id = "/base_link"
    # define shape
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    # define scale
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    # define color
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    # define position
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.y = z

    return marker

def callback(yolo, depth_msg, k_msg):
    """
    Callback function. Computes depth of detected objects and appends markers for object position to the global MarkerArray
         @param yolo: ROS Message containing the bounding boxes of detected objects
         @param depth_msg: ROS Message containing the depth map
         @param k_msg: ROS Message containing camera info
    """
    global markerArray

    # empty markerArray with DELETEALL marker
    markerArray = MarkerArray()
    markerArray.markers.append(delete_marker())

    # get the calibration matrix from the camera info message and extract information
    K = k_msg.K # calibration matrix

    # prinicpal point
    cx= K[2]
    cy = K[5]
    # focal point
    fx = K[0]
    fy = K[4]

    # get the bounding boxes
    bb = yolo.bounding_boxes

    # convert pixel values of depth map into real depth
    bridge = CvBridge()
    depths = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    # loop over every bounding box
    for b in bb:

        # only compte depth for objects with a high certainty
        if  b.probability >= 0.7:

            # get height and width of bounding box
            h = b.ymax - b.ymin # height
            w = b.xmax - b.xmin # width

            # compte center coordinate of bounding box
            u = b.xmin + w/2
            v = b.ymin + h/2

            # get depth of object
            depth = depths[v][u]


            p = 1 # counter
            # while the depth is NaN move 5 pixels right and down and keep depth that is not NaN. Stop if the index is out of bounds.
            while math.isnan(depth) and p <= 5 and v+p < 1280 and u+p < 720:
                if not math.isnan(depths[v+p][u]):
                    depth = depths[v+p][u]

                elif not math.isnan(depths[v][u+p]):
                    depth = depths[v][u+p]
                else:
                    depth = depths[v+p][u+p]
                    p = p+1

            # compute real world coordinate of X and Y
            X = (u-cx)/fx * depth
            Y = (v-cy)/fy * depth

            # if the depth is not NaN define a marker and append to the list
            if not math.isnan(depth):
                markerArray.markers.append(make_marker(depth,  Y, -X, 0.2))

            rospy.loginfo('Object: {} --- Depth: {}'.format(b.Class, depth))

    print('\n\n')
    

def listener():
    global markerArray

    # init depth_yolo node
    rospy.init_node('depth_yolo', anonymous=True)

    # define topic for publishinf
    topic = "visualization_marker_array"

    # define publisher
    pub = rospy.Publisher(topic, MarkerArray)

    # subscribe to bonding boxes from darknet_ros
    yolo_sub = message_filters.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes)
    # subsribe to the depth_map from zed2
    depth_sub = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image)
    # subcribe to the camera info from zed2
    k_sub = message_filters.Subscriber('/zed2/zed_node/depth/camera_info', CameraInfo)

    # If all messages arrive approximately at the same time compute the 3D coordinates
    ts = message_filters.ApproximateTimeSynchronizer([yolo_sub, depth_sub, k_sub], 100, 1)
    ts.registerCallback(callback)

    # while the node is running publish the markerArray
    while not rospy.is_shutdown():
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        pub.publish(markerArray)

        rospy.sleep(0.01)

if __name__ == '__main__':
    listener()

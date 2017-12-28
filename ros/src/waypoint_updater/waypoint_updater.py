#!/usr/bin/env python

import rospy
import tf
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from sensor_msgs.msg import Image

import math
import numpy as np

# IMPORTS TO DELETE 
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

REF_VELOCITY = 4.5 

MAX_DECEL = 1.0

class WaypointUpdater(object):
	def __init__(self):
		rospy.init_node('waypoint_updater')

		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
		self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
		# TODO Later we will need this subscribers, not now
		rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
		#rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

		#rospy.Subscriber('/image_color', Image, self.image_cb)

		self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


		# TODO: Add other member variables you need below
		self.pose_x = 0
		self.pose_y = 0
		self.pose_z = 0
		self.yaw = 0
		self.pitch = 0
		self.roll = 0
                self.waypoints = []
		self.max_velocity = 0.0
		self.traffic_light_index = -1
		self.bridge = CvBridge()
		
		rospy.loginfo("*****************PATH:" + os.path.dirname(os.path.abspath(__file__)))
		
		self.loop()


	def loop(self):
	    rate = rospy.Rate(50) # Spin 50Hz

	    # TODO(jason): puts some buffer to make sure the car stops behind
	    # the line.  Possibly should improve the stop line waypoint
	    # selection
	    STOP_LINE_OFFSET = 5
	    while not rospy.is_shutdown():
		index = self.get_closest_waypoints()
		if index == -1:
		    # NOTE(jason): various problems when we have don't
		    # have a valid position so just skip until we do
		    rate.sleep()
		    continue

		end_index = len(self.waypoints) - 1

		# NOTE(jason): clamp to end of waypoints
		last_index = min(index + LOOKAHEAD_WPS, end_index)

		rospy.loginfo("index: %s, traffic light: %s", index, self.traffic_light_index)

		next_waypoints = Lane()
		next_waypoints.header.stamp = rospy.Time(0)

		stop_index = -1
		if self.traffic_light_index != -1:
		    stop_index = self.traffic_light_index - STOP_LINE_OFFSET

		next_waypoints.waypoints = []
		for i in range(index, last_index + 1):
		    wp = self.waypoints[i]
		    p = Waypoint()
		    p.pose = wp.pose
		    if stop_index == -1:
			p.twist.twist.linear.x = wp.twist.twist.linear.x
		    else:
			d = self.distance(self.waypoints, i, stop_index)
			v = math.sqrt(2*MAX_DECEL*d)
			p.twist.twist.linear.x = min(v, wp.twist.twist.linear.x)

		    next_waypoints.waypoints.append(p)

		self.final_waypoints_pub.publish(next_waypoints)

		rate.sleep()

	def pose_cb(self, msg):
		self.pose_x = msg.pose.position.x
		self.pose_y = msg.pose.position.y
		self.pose_z = msg.pose.position.z # We don't need z position I think

		orientation = msg.pose.orientation
		# about the orientation: we only need yaw for this practice since going 10km/h will never make 	the car drift ;)
		euler = tf.transformations.euler_from_quaternion(
			[orientation.x,
			orientation.y,
			orientation.z,
			orientation.w])
		self.roll = euler[0]
		self.pitch = euler[1]
		self.yaw = euler[2]



	def waypoints_cb(self, waypoints):
		self.waypoints = waypoints.waypoints
		self.max_velocity = self.get_waypoint_velocity(self.waypoints[40])
		# Maybe we can unsubscribe from this node since we don't need it anymore
		self.base_waypoints_sub.unregister()

	def traffic_cb(self, msg):
		# TODO: Callback for /traffic_waypoint message. Implement
                self.traffic_light_index = msg.data

	def obstacle_cb(self, msg):
		# TODO: Callback for /obstacle_waypoint message. We will implement it later
		pass
		
	def image_cb(self, msg):
		try:
			rospy.loginfo("*****************PATH:" + os.getcwd())
			cv2_img = self.bridge.imgmsg_to_cv2( msg, 'bgr8')
			n = msg.header.seq
			
		except CvBridgeError as e: 
			rospy.logwarn(e)
		else:
			folder = './images/image{}.png'.format(n)
			cv2.imwrite(folder, cv2_img)

	def get_closest_waypoints(self):
		# To get the closest waypoint we will use self.pose_x, self.pose_y, self.yaw and self.waypoints 
		closest_distance = 1000000
		closest_point = -1
		for i in range(len(self.waypoints)):
			wp_x = self.waypoints[i].pose.pose.position.x
			wp_y = self.waypoints[i].pose.pose.position.y
			
			distance = math.sqrt((self.pose_x - wp_x)**2 + (self.pose_y - wp_y)**2)

			# Since we want the closest waypoint ahead, we need to calculate the angle between the car 				and the waypoint
			psi = np.arctan2(self.pose_y - wp_y, self.pose_x - wp_x)
			dtheta = np.abs(psi - self.yaw)
 
			# NOTE(jason): it seems like the yaw check might be
			# missing the case when we're at the waypoint
			#if (distance < closest_distance and dtheta < np.pi/4) :
			if distance < closest_distance:
				closest_distance = distance
				closest_point = i

		return closest_point

			

	def get_waypoint_velocity(self, waypoint):
		return waypoint.twist.twist.linear.x

	def set_waypoint_velocity(self, waypoints, waypoint, velocity):
		waypoints[waypoint].twist.twist.linear.x = velocity

	def distance(self, waypoints, wp1, wp2):
		dist = 0
		dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
		for i in range(wp1, wp2+1):
			dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
			wp1 = i
		return dist


if __name__ == '__main__':
	try:
		WaypointUpdater()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start waypoint updater node.')

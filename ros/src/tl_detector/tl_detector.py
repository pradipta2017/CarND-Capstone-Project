#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
        pose (Pose): position to match a waypoint to

        Returns:
        int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_distance = 1000000
        closest_point = -1
        pose_x = pose.position.x
        pose_y = pose.position.y
        orientation = pose.orientation
        euler = tf.transformations.euler_from_quaternion(
        [orientation.x,
        orientation.y,
        orientation.z,
        orientation.w])
        yaw = euler[2]
        if self.waypoints is not None:
	
            for i in range(len(self.waypoints.waypoints)):
                wp_x = self.waypoints.waypoints[i].pose.pose.position.x
                wp_y = self.waypoints.waypoints[i].pose.pose.position.y
			
                distance = math.sqrt((pose_x - wp_x)**2 + (pose_y - wp_y)**2)

                # Since we want the closest waypoint ahead, we need to calculate the angle between the car and the waypoint
                psi = np.arctan2(pose_y - wp_y, pose_x - wp_x)
                dtheta = np.abs(psi - yaw)

                if (distance < closest_distance and dtheta < np.pi/4) :
                    closest_distance = distance
                    closest_point = i

        return closest_point

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
	"""Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
	stop_line_positions = self.config['stop_line_positions']

        #TODO find the closest visible traffic light (if one exists)
	stop_at = 999999

	if self.waypoints is not None:
		car_x = self.pose.pose.position.x
		car_y = self.pose.pose.position.y
		
		for each in self.lights:
			light_x = each.pose.pose.position.x
			light_y = each.pose.pose.position.y
			if(light_x < stop_at):
				if(light_x > car_x):
					stop_at = light_x
					#Need Attention: sometime y value of traffic light is not alligned with waypoint y
					# car_y is assigned to get an approximation
					each.pose.pose.position.y = car_y
					light = each
				

        '''
        next_stop_x, next_stop_y will give the (x,y) coordinate of next traffic light.
        We need to set the 'light' variable to True at a safe distance from where car can slow-down/stop.
        '''
        if light:
	    light_wp = self.get_closest_waypoint(light.pose.pose)
	    state = light.state
            return light_wp, state
        #self.waypoints = None #Need Attention: Not sure why it was set to None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

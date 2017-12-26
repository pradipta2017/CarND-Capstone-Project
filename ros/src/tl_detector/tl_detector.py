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
        #rospy.loginfo("traffic light config: %s", self.config)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.stop_line_waypoints = []

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # NOTE(jason): Pre-calculating the stop line positions so the check on
        # each frame will hopefully do less work by only checking the
        # stop_line_waypoints list for a close waypoint index
	stop_line_positions = self.config['stop_line_positions']
        sl_wp = []
        for slp in stop_line_positions:
            best_d = 10
            sli = -1
            for i, wp in enumerate(self.waypoints.waypoints):
                wpx = wp.pose.pose.position.x
                wpy = wp.pose.pose.position.y
                d = math.sqrt((wpx - slp[0])**2 + (wpy - slp[1])**2)
                if d < best_d:
                    best_d = d
                    sli = i

            # not a waypoint near the stop line.  This is really just for the
            # parking lot scenario that doesn't have any lights.
            if sli != -1:
                sl_wp.append(sli)

        rospy.loginfo("sl_wp: %s", sl_wp)

        self.stop_line_waypoints = sl_wp


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

        light_index = -1
        light_state = TrafficLight.UNKNOWN
        if self.pose:
            car_index = self.get_closest_waypoint(self.pose.pose)

            # NOTE(jason): Look for the stop line waypoint index in ahead
            # within the waypoint size sent in waypoint_updater.
            # TODO(jason): This may need to be a shorter range if the lights
            # are still too small for the classifier to tell what color they
            # are.  It seems like they're still pretty small in the simulator
            # when it detects them.
            for i in self.stop_line_waypoints:
                if i > car_index and i - car_index < 100:
                    light_index = i
                    break

            if light_index != -1:
                # TODO(jason): start using this instead when the classifier is
                # working
                # light_state = self.get_light_state()

                # there's actually a light so find closest light to waypoint in
                # order to get color.
                sl_wp = self.waypoints.waypoints[light_index]
                dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
                best_d = 100
                for l in self.lights:
                    d = dl(sl_wp.pose.pose.position, l.pose.pose.position)
                    if d < best_d:
                        best_d = d
                        light = l
                light_state = light.state


        return light_index, light_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

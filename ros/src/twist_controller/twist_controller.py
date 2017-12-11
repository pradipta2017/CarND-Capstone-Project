import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
# Time between two meassurements = 1/Hz
DELTA_T = 1/50

# Maybe implement twiddle
P_VEL = 1.
I_VEL = 0.
D_VEL = 1.

P_THR = 1.
I_THR = 0.
D_THR = 1.

TAU = 0
TS = 0


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        
        self.linear_velocity = kwargs['linear_velocity']
        self.angular_velocity = kwargs['angular_velocity']
        self.current_velocity = kwargs['curr_velocity']
        
        self.vehicle_mass = kwargs['vehicle_mass']
		self.fuel_capacity = kwargs['fuel_capacity']
		self.brake_deadband = kwargs['brake_deadband']
		self.decel_limit = kwargs['decel_limit']
		self.accel_limit = kwargs['accel_limit']
		self.wheel_radius = kwargs['wheel_radius']
		self.wheel_base = kwargs['wheel_base']
		self.steer_ratio = kwargs['steer_ratio']
		self.max_lat_accel = kwargs['max_lat_accel']
		self.max_steer_angle = kwargs['max_steer_angle']
		
		# PID for linear velocity
		self.pid_velocity = PID(P_VEL, I_VEL, D_VEL, self.decel_limit, self.accel_limit)
		
		# PID for throttle value
		self.pid_throttle = PID(P_THR, I_THR, D_THR, 0, 1)
		
		# I don't know how to use the lowpass filter, if anyone does go for it
		self.low_pass_filter = LowPassFilter(TAU, TS)
		
		# Brake torque, I got the info of how to calculate the torke in this page https://sciencing.com/calculate-brake-torque-6076252.html
		total_vehicle_mass = self.vehicle_mass + self.fuel_capacity/GAS_DENSITY
        diff_vel = self.current_velocity - self.prop_velocity
        newtons = total_vehicle_mass * (diff_vel/DELTA_T)
        self.brake_torque = newtons / self.wheel_radius
        
        # Steering controller
        self.yaw_controller = YaWController(self.wheel_base, self.steer_ratio, self.min_speed, self.max_lat_accel, self.max_steer_angle)
        

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        steering = self.yaw_controller.get_steering(self.linear_velocity, self.angular_velocity, self.current_velocity)
        if steering > max_steer_angle: 
        	# In order keep the lane, if the desired steering angle is > than max_steer_angle, we will have to slow down and set the steering to the maximum or slow down until steering < max_steer_angle
        	steering = max_steer_angle
        
        
        return 0., brake, steering

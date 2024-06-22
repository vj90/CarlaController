#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self.current_ref_pt      = None
        self.lookahead           = 0 #m
        self.last_min_idx_ref_pt = 0
        self.KIv                 = 0.01    # Integral control for v
        self.KPv                 = 1  # Proportional control for v
        self.KDv                 = 0.01    # Derivative control for v
        self.kVs                 = 4.8    # Constant for Stanley controller
        self.KDelta              = 1    # Constant for steering output
        self.cte                 = None # cross-track error
        self.ref_psi             = None # ref yaw angle
        self.xFA                 = 0 # x at front axle
        self.yFA                 = 0 # y at front axle

    def cgToFA(self):
        lf = 1.5
        self.xFA = self._current_x + lf*np.cos(self._current_yaw)
        self.yFA = self._current_y + lf*np.sin(self._current_yaw)

    def getMinDistFromVector(self,x1,y1,x2,y2):
        x = self.xFA
        y = self.yFA
        # b is the vector from x1,y1 to x2,y2
        b = np.array([x2-x1,y2-y1])
        # a is the vector from x1,y1 to x,y
        a = np.array([x-x1,y-y1])
        len_b = np.linalg.norm(b)
        if len_b<0.0001: # b is a point, return the length of a
            assert False, "Waypoints should not coincide"
        else:
            # if b cross a is positive, a lies on the left of b
            return np.cross(b,a)/len_b
        
    def ref_yaw(self,x1,y1,x2,y2):
        return np.arctan2(y2-y1,x2-x1)

    def getMinDistanceFromWaypoints(self):
        assert self.last_min_idx_ref_pt>=0,"Closest waypoint not set"
        assert self.last_min_idx_ref_pt<len(self._waypoints),"Index out of bounds in getMinDistanceFromWaypoints"
        prev_k = self.last_min_idx_ref_pt
        next_k = self.last_min_idx_ref_pt+1
        if self.last_min_idx_ref_pt>len(self._waypoints)-2:
            prev_k = self.last_min_idx_ref_pt -1
            next_k = self.last_min_idx_ref_pt
        self.ref_psi = self.ref_yaw(self._waypoints[prev_k][0],self._waypoints[prev_k][1],self._waypoints[next_k][0],self._waypoints[next_k][1])
        self.cte = self.getMinDistFromVector(self._waypoints[prev_k][0],self._waypoints[prev_k][1],self._waypoints[next_k][0],self._waypoints[next_k][1])

    def updateCTE(self):
        min_idx_ref_pt      = 0
        min_dist_ref        = float("inf")
        for i in range(min_idx_ref_pt,len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self.xFA,
                    self._waypoints[i][1] - self.yFA]))
            # only start looking for lookahead after having found the min dist point
            if abs(dist-self.lookahead) < min_dist_ref:
                min_dist_ref = dist
                min_idx_ref_pt = i
        if min_idx_ref_pt < len(self._waypoints)-1:
            self.last_min_idx_ref_pt = min_idx_ref_pt
        else:
            self.last_min_idx_ref_pt = len(self._waypoints)-1
        self.getMinDistanceFromWaypoints()
        self.current_ref_pt = [self._waypoints[self.last_min_idx_ref_pt][0],self._waypoints[self.last_min_idx_ref_pt][1]]

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(min_idx,len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
            min_idx = -1
        self._desired_speed = desired_speed
        
    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def debugOutput(self):
        print("CTE = ", self.cte)
        print("degree error = ", str(self._current_yaw - self.ref_psi))

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        self.cgToFA()
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0
        self.updateCTE()
        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('v_error_integral',0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """
            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            # pid controller
            # assuming that the constants take care of dt
            error_ref_v_p = self._desired_speed - v
            error_ref_v_i = self.vars.v_error_integral
            self.vars.v_error_integral = self.vars.v_error_integral + error_ref_v_p
            error_ref_v_d = error_ref_v_p - (self._desired_speed - self.vars.v_previous)  # current error - prev error
            acc_output = self.KPv*error_ref_v_p + self.KIv*error_ref_v_i + self.KDv*error_ref_v_d
            

            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            if acc_output > 0:
                throttle_output = acc_output
            else:
                brake_output    = acc_output

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            eps = 0.01
            delta_yaw = self._current_yaw - self.ref_psi
            delta = 0.5*delta_yaw + 0.08*np.arctan2(self.kVs*self.cte,self._current_speed+eps)
            
            # Change the steer output with the lateral controller. 
            # Positive --> steer to the right
            # Also, somehow everything is inverted ??
            # --> error negative --> too far left --> steer right
            steer_output    = -1*self.KDelta*delta

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)
            self.debugOutput()
        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step

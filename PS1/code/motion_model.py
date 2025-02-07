'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # Effect of rotation on rotational variance
        self._alpha1 = 0.0005

        # Effect of translation on rotational variance
        self._alpha2 = 0.0005

        # Effect of translation on translation variance
        self._alpha3 = 0.005

        # Effect of rotation on translation variance
        self._alpha4 = 0.005


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """

        # Calculate motion from odometry
        d_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        d_trans = math.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)
        d_rot2 = u_t1[2] - u_t0[2] - d_rot1

        # Add noise
        D_rot1 = d_rot1 - np.random.normal(0, np.sqrt(self._alpha1 * d_rot1**2 + self._alpha2 * d_trans**2))
        D_trans = d_trans - np.random.normal(0, np.sqrt(self._alpha3 * d_trans**2 + self._alpha4 * d_rot1**2 + self._alpha4 * d_rot2**2))
        D_rot2 = d_rot2 - np.random.normal(0, np.sqrt(self._alpha1 * d_rot2**2 + self._alpha2 * d_trans**2))

        # Calculate state after noise
        x = x_t0[0] + D_trans * math.cos(x_t0[2] + D_rot1)
        y = x_t0[1] + D_trans * math.sin(x_t0[2] + D_rot1)
        theta = x_t0[2] + D_rot1 + D_rot2

        x_t1 = np.array([x, y, theta])
        return x_t1

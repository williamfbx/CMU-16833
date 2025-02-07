'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 5
        self._z_short = 0.2
        self._z_max = 1
        self._z_rand = 250

        self._sigma_hit = 50
        self._lambda_short = 0.1
        self._occupancy_map = occupancy_map

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        prob_zt1 = 1.0
        log_prob_zt1 = 0.0
        for k in range(0, 180, self._subsampling):

            # Given particle state belief, use ray casting to find the expected laser range reading
            z_t1 = z_t1_arr[k]
            theta = x_t1[2] + np.deg2rad(k-90)

            # Lazer is offset by 25 cm
            x = x_t1[0] + 25 * np.cos(x_t1[2])
            y = x_t1[1] + 25 * np.sin(x_t1[2])

            z_t1s = self._max_range
            for i in range(0, self._max_range, 10):
                x_occ = x + i * np.cos(theta)
                y_occ = y + i * np.sin(theta)
                x_occ = int(x_occ/10)
                y_occ = int(y_occ/10)

                # Lazer exceeds map boundary
                if x_occ >= 800 or y_occ >= 800:
                    z_t1s = self._max_range
                    break
                elif x_occ < 0 or y_occ < 0:
                    z_t1s = self._max_range
                    break

                # Lazer hits something
                elif self._occupancy_map[y_occ, x_occ] > self._min_probability:
                    z_t1s = i
                    break

            # Normal distribution
            if z_t1 > self._max_range or z_t1 < 0:
                p_z_hit = 0
            else:
                p_z_hit = self._z_hit * norm.pdf(z_t1, z_t1s, self._sigma_hit)

            # Exponential distribution
            if z_t1 < z_t1s and z_t1 > 0:
                p_short = self._z_short * self._lambda_short * np.exp(-self._lambda_short * z_t1)
            else:
                p_short = 0
            
            # Max range distribution
            if z_t1 >= self._max_range:
                p_max = self._z_max * 1
            else:
                p_max = 0

            # Unexplainable measurements distribution
            if z_t1 < self._max_range and z_t1 >= 0:
                p_rand = self._z_rand / self._max_range
            else:
                p_rand = 0
            
            p = p_z_hit + p_short + p_max + p_rand
            prob_zt1 *= p
            log_prob_zt1 += np.log(max(p, 1e-10))


        # return prob_zt1
        return np.exp(log_prob_zt1)
'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def visualize_lidar_scan(laser_pose, ranges, tstep, output_path):
    """
    Visualizes the lidar scan in its own figure. The sensor is placed at the center,
    and the beams are rotated to match the sensor's heading as given by odometry.
    
    Parameters:
      laser_pose : [x, y, theta] for the laser sensor (world frame)
      ranges     : list or array of range measurements (assumed 180 beams spanning -90° to +90° relative to the sensor)
      tstep      : timestep number (for labeling/saving the figure)
      output_path: directory to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a new figure for the lidar scan
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Lidar Scan at Timestep {:04d}".format(tstep))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # Set limits (adjust these if needed)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True)

    # The sensor is at the origin of the scan view.
    sensor_x = 0.0
    sensor_y = 0.0

    # Assume the lidar provides 180 beams spanning from -90° to +90°.
    num_beams = len(ranges)
    # Create an array of beam angles in radians (relative to the sensor’s local frame)
    relative_angles = np.deg2rad(np.linspace(-90, 90, num_beams))
    
    # Use the sensor's heading from odometry (laser_pose[2]) to rotate the scan.
    sensor_heading = laser_pose[2]
    cos_theta = np.cos(sensor_heading)
    sin_theta = np.sin(sensor_heading)
    # Rotation matrix for the sensor's heading:
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta,  cos_theta]])

    # For each beam, compute its endpoint in the sensor's frame (after applying rotation)
    # Note: In your other parts of the code you divide by 10 to convert from world units to map units.
    # Here, we'll assume that dividing by 10 converts the range into meters for display.
    for r, angle in zip(ranges, relative_angles):
        # Convert the range to meters (adjust if your units differ)
        r_m = r / 10.0  
        # Compute the beam's endpoint in the sensor's local coordinates.
        # In the sensor's own coordinate frame, 0° is straight ahead.
        local_endpoint = np.array([r_m * np.cos(angle),
                                   r_m * np.sin(angle)])
        # Rotate the endpoint by the sensor's heading.
        rotated_endpoint = R @ local_endpoint
        x_end, y_end = rotated_endpoint

        # Plot the beam as a line from (0,0) to (x_end, y_end)
        ax.plot([sensor_x, x_end], [sensor_y, y_end], 'b-', linewidth=0.5)

    # Mark the sensor location
    ax.plot(sensor_x, sensor_y, 'ro', markersize=5)

    # Draw an arrow to indicate the sensor's forward direction.
    arrow_length = 1.0  # in meters; adjust as needed
    forward_endpoint = R @ np.array([arrow_length, 0])
    ax.arrow(sensor_x, sensor_y,
             forward_endpoint[0], forward_endpoint[1],
             head_width=0.2, head_length=0.3, fc='r', ec='r')

    # Save (or show) the figure.
    filename = "{}/{:04d}_lidar.png".format(output_path, tstep)
    plt.savefig(filename)
    plt.close(fig)

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))
    for i in range(num_particles):
        while True:
            x = np.random.uniform(0, 8000)
            y = np.random.uniform(0, 8000)
            if occupancy_map[int(y/10), int(x/10)] == 0:
                X_bar_init[i, 0] = x
                X_bar_init[i, 1] = y
                X_bar_init[i, 2] = np.random.uniform(-3.14, 3.14)
                X_bar_init[i, 3] = 1.0 / num_particles
                break
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata3.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--output_lidar', default='results_lidar')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize_lidar', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    #X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        # X_bar = resampler.low_variance_sampler(X_bar)
        # X_bar = resampler.multinomial_sampler(X_bar)
        # X_bar = resampler.dynamic_multinomial_sampler(X_bar)
        X_bar = resampler.dynamic_low_variance_sampler(X_bar)
        num_particles = X_bar.shape[0]

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)

            if args.visualize_lidar:
                if (meas_type == "L"):
                    visualize_lidar_scan(odometry_laser, ranges, time_idx, args.output_lidar)
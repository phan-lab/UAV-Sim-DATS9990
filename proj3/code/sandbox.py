import csv
import inspect
import numpy as np
import time
import random
from pathlib import Path

import matplotlib.pyplot as plt

from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World
from flightsim.crazyflie_params import quad_params
from flightsim.sensors.vio_utils import Vio
from flightsim.sensors.stereo_utils import StereoUtils
from ml_models import imu_buffer, predictions, inference_times

#######################################################################
from proj3.code.se3_control import SE3Control
from proj3.code.world_traj import WorldTraj
#######################################################################

# Load the test example.
filename = 'test_maze.json'
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
# resolution = world.world['resolution'] # (x,y,z) resolution of discretization, shape=(3,).
# margin = world.world['margin']         # Scalar spherical robot size or safety margin.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()

planning_end_time = time.time()


# Set simulation parameters.
t_final = 15
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}
print("initial_state = ", initial_state)

visualize_stereo_features = False
if visualize_stereo_features:
    plt.show()
    pass

# maximum number of features considered for VIO, increasing it will make VIO more robust, but the less efficient
max_num_features = 150
# feature sample resolution (in meter), increasing it will make VIO more efficient, but the less robust
sample_resolution = 1.25


# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.
print()
# print('Simulate.')
# (sim_time, state, est_state, control, flat, exit, imu_measurements) = simulate(initial_state,
#                                               quadrotor,
#                                               my_se3_control,
#                                               my_world_traj,
#                                               t_final, stereo=stereo, vio=vio)
# print(exit.value)

N_RUNS = 10

def generate_thrust():
    r = random.random()
    if r < 0.65:
        randomized_thrust = random.uniform(0.95, 1.0)
    elif r < 0.9:
        randomized_thrust = random.uniform(0.65, 0.85)
    else:
        randomized_thrust = random.uniform(0.3, 0.5)
    # # randomized_thrust = random.uniform(0.0, 0.3)
    # # return random.uniform(0.05, 0.25)
    return randomized_thrust
    # return random.uniform(0.3, 0.7)

# rotor_indices = [-1, 0, 1, 2, 3]
# broken_index = random.choice(rotor_indices)
#
# thrust_scale = random.uniform(0.0, 1.0)

with open("imu_readings.csv", "w", newline='') as data_file:
    writer = csv.writer(data_file)

    # Binary Model
    # writer.writerow(['run_id', 'time', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z', 'broken_index', 'thrust_scale', 'fault_active'])

    writer.writerow(
        ['run_id', 'time', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y',
             'gyroscope_z', 'rotor0_eff', 'rotor1_eff', 'rotor2_eff', 'rotor3_eff'])

    for run_id in range(N_RUNS):
        # reset model parameters for fresh run
        imu_buffer.clear()
        predictions.clear()
        inference_times.clear()

        my_se3_control.reset()

        vio = Vio()
        stereo = StereoUtils(world, vio.camera_matrix, sample_resolution=sample_resolution,
                             visualization=visualize_stereo_features, max_num_features=max_num_features)

        my_world_traj = WorldTraj(world, start, goal)

        # Binary Model
        # rotor_indices = [-1, 0, 1, 2, 3]
        # broken_index = random.choice(rotor_indices)
        # thrust_scale = 1.0

        fault_active = random.choice([0, 1])
        # fault_active = 0
        thrust_scale = []
        fault_profile = "normal"

        # Binary Model
        # if broken_index != -1:
        #     thrust_scale = generate_thrust()
        #     fault_profile = random.choice(["abrupt", "ramp", "intermittent"])
        # else:
        #     thrust_scale = random.uniform(0.85, 1.15)

        if fault_active == 1:
            for index in range(0, 4):
                thrust_scale.append(generate_thrust())
            fault_profile = random.choice(["abrupt", "ramp", "intermittent"])
        else:
            thrust_scale.extend([1.0] * 4)

        # Binary Model
        # fault_time = round(random.uniform(0.1 * t_final, 0.8 * t_final), 3) if broken_index != -1 else 100

        fault_time = round(random.uniform(0.1 * t_final, 0.6 * t_final), 3) if fault_active == 1 else 100

        print('Simulate.')
        print(thrust_scale, fault_profile, fault_time)

        # Binary Model
        # (sim_time, state, est_state, control, flat, exit, imu_measurements) = simulate(initial_state,
        #                                                                                quadrotor,
        #                                                                                my_se3_control,
        #                                                                                my_world_traj,
        #                                                                                t_final, stereo=stereo, vio=vio,
        #                                                                                broken_index=broken_index,
        #                                                                                thrust_scale=thrust_scale,
        #                                                                                fault_time=fault_time,
        #                                                                                fault_profile=fault_profile)

        (sim_time, state, est_state, control, flat, exit, imu_measurements) = simulate(initial_state,
                                                                                       quadrotor,
                                                                                       my_se3_control,
                                                                                       my_world_traj,
                                                                                       t_final, stereo=stereo, vio=vio,
                                                                                       thrust_scale=thrust_scale,
                                                                                       fault_time=fault_time,
                                                                                       fault_profile=fault_profile)

        print(exit.value)

        if exit.value in (ExitStatus.DEGRADED_SAFE_LAND.value, ExitStatus.EMERGENCY_LAND.value):
            continue

        accelerometer_data = [imu[0] for imu in imu_measurements]
        gyroscope_data = [imu[1] for imu in imu_measurements]

        for i in range(len(accelerometer_data)):
                accelerometer_x, accelerometer_y, accelerometer_z = accelerometer_data[i]
                gyroscope_x, gyroscope_y, gyroscope_z = gyroscope_data[i]
                time_stamp = sim_time[i]

                # imu = np.array([accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z], dtype=np.float32)
                # eff_pred, t_inf = process_imu_and_predict(time_stamp, imu)

                writer.writerow([run_id,
                                time_stamp,
                                accelerometer_x,
                                accelerometer_y,
                                accelerometer_z,
                                gyroscope_x,
                                gyroscope_y,
                                gyroscope_z,
                                1.0 if time_stamp < fault_time else thrust_scale[0],
                                1.0 if time_stamp < fault_time else thrust_scale[1],
                                1.0 if time_stamp < fault_time else thrust_scale[2],
                                1.0 if time_stamp < fault_time else thrust_scale[3]])

                # Binary Model
                # writer.writerow([run_id,
                #                 time_stamp,
                #                 accelerometer_x,
                #                 accelerometer_y,
                #                 accelerometer_z,
                #                 gyroscope_x,
                #                 gyroscope_y,
                #                 gyroscope_z,
                #                 # 0 if time_stamp < fault_time else (broken_index + 1)])
                #                 -1 if time_stamp < fault_time else broken_index,
                #                 1 if time_stamp < fault_time else thrust_scale,
                #                 0 if time_stamp < fault_time else 1])

        collision_pts = world.path_collisions(state['x'], robot_radius)

        # increase the goal reached tolerance for VIO noisy state estimation and accumulated drift
        goal_tolerance = 2
        stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= goal_tolerance

        no_collision = collision_pts.size == 0
        flight_time = sim_time[-1]
        flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0), axis=1))
        planning_time = planning_end_time - planning_start_time

        print()
        print(f"Results:")
        print("Run result")
        print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
        print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
        print(f"  Flight time:     {flight_time:.1f} seconds")
        print(f"  Flight distance: {flight_distance:.1f} meters")
        # print(sum(inference_times) / len(inference_times))

###############VIO PLOTTING####################################
# %% Gather results
n = len(vio.pose)

euler = np.zeros((n, 3))
translation = np.zeros((n, 3))
velocity = np.zeros((n, 3))
a_bias = np.zeros((n, 3))

for (i, p) in enumerate(vio.pose):
    euler[i] = p[0].as_euler('XYZ', degrees=True)
    translation[i] = p[1].ravel()
    velocity[i] = p[2].ravel()
    a_bias[i] = p[3].ravel()

###############PLANNING PLOTTING##############################

# plot state vs vio_state
# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.
if exit.value not in (ExitStatus.DEGRADED_SAFE_LAND.value, ExitStatus.EMERGENCY_LAND.value):
    collision_pts = world.path_collisions(state['x'], robot_radius)

    # increase the goal reached tolerance for VIO noisy state estimation and accumulated drift
    goal_tolerance = 2
    stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= goal_tolerance

    no_collision = collision_pts.size == 0
    flight_time = sim_time[-1]
    flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
    planning_time = planning_end_time - planning_start_time
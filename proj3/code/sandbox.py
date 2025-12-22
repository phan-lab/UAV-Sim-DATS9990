import csv
import inspect
import numpy as np
import pandas as pd
import time
import random
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World
from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.sensors.vio_utils import Vio
from flightsim.sensors.stereo_utils import StereoUtils

#######################################################################
from proj3.code.se3_control import SE3Control
from proj3.code.world_traj import WorldTraj
#######################################################################

class RotorEffectivenessPredictor(nn.Module):
    """
    Predicts effectiveness per rotor
    """
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        last_hidden = lstm_out.mean(dim=1)
        out = self.fc(last_hidden)
        return out


class RotorEffectivenessPredictor2(nn.Module):
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )

    def forward(self, x, hc=None):
        x = x.permute(0, 2, 1)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)

        if hc is None:
            lstm_out, _ = self.lstm(cnn_out)
        else:
            lstm_out, hc = self.lstm(cnn_out, hc)

        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


class BinaryFaultClassifier(nn.Module):
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # self.lstm = nn.LSTM(
        #     input_size=128,
        #     hidden_size=hidden_size,
        #     num_layers=lstm_layers,
        #     batch_first=True,
        #     dropout=dropout if lstm_layers > 1 else 0,
        # )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            # nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = x.transpose(1, 2)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        # weights = self.attn(lstm_out)
        # context = (lstm_out * weights).sum(dim=1)
        # out = self.fc(context)
        # last_hidden = lstm_out[:, -1, :]
        last_hidden = lstm_out.mean(dim=1)
        out = self.fc(last_hidden)
        return out.squeeze(1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regression model
# model = RotorEffectivenessPredictor().to(DEVICE)
# model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
# model.eval()

# Regression model - Unidirectional
model = RotorEffectivenessPredictor2().to(DEVICE)
model.load_state_dict(torch.load("best_model2.pt", map_location=DEVICE))
model.eval()

# Binary model
# model = BinaryFaultClassifier().to(DEVICE)
# model.load_state_dict(torch.load("best_binary_model.pt", map_location=DEVICE))
# model.eval()
# -----------------------------
WINDOW_LEN = 100
imu_buffer = []      # will store last 100 IMU readings
predictions = []     # list of (time, predicted effectiveness vector)
inference_times = [] # list of inference timing results

print("Model loaded and ready for inference!")

def process_imu_and_predict(curr_time, imu_sample):
    imu_buffer.append(imu_sample)

    if len(imu_buffer) > WINDOW_LEN:
        imu_buffer.pop(0)

    # Run prediction only when window is full
    if len(imu_buffer) == WINDOW_LEN:
        imu_window = np.array(imu_buffer, dtype=np.float32)

        # (1, W, 6) tensor
        x = torch.tensor(imu_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(x).cpu().numpy()[0]
        t_inf = (time.perf_counter() - start) * 1000  # ms

        predictions.append((curr_time, pred))
        inference_times.append(t_inf)

        print(f"t={curr_time:.2f}s  eff={pred}  inference={t_inf:.3f}ms")

        return pred, t_inf

    return None, None

np.random.seed(0)

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
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()


# Set simulation parameters.
t_final = 15
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}
print("initial_state = ", initial_state)


vio = Vio()

visualize_stereo_features = False
if visualize_stereo_features:
    plt.show()
    pass

# maximum number of features considered for VIO, increasing it will make VIO more robust, but the less efficient
max_num_features = 150
# feature sample resolution (in meter), increasing it will make VIO more efficient, but the less robust
sample_resolution = 1.25
stereo = StereoUtils(world, vio.camera_matrix, sample_resolution = sample_resolution, visualization=visualize_stereo_features, max_num_features=max_num_features)


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

N_RUNS = 1

def generate_thrust():
    r = random.random()
    if r < 0.5:
        randomized_thrust = random.uniform(0.0, 0.25)
    elif r < 0.8:
        randomized_thrust = random.uniform(0.25, 0.6)
    else:
        randomized_thrust = random.uniform(0.6, 1.0)
    # randomized_thrust = random.uniform(0.0, 0.3)
    # return random.uniform(0.05, 0.25)
    return randomized_thrust

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
        # Binary Model
        # rotor_indices = [-1, 0, 1, 2, 3]
        # broken_index = random.choice(rotor_indices)
        # thrust_scale = 1.0

        fault_active = random.choice([0, 1])
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

        fault_time = round(random.uniform(0.1 * t_final, 0.8 * t_final), 3) if fault_active == 1 else 100

        print('Simulate.')

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

        accelerometer_data = [imu[0] for imu in imu_measurements]
        gyroscope_data = [imu[1] for imu in imu_measurements]

        for i in range(len(accelerometer_data)):
                accelerometer_x, accelerometer_y, accelerometer_z = accelerometer_data[i]
                gyroscope_x, gyroscope_y, gyroscope_z = gyroscope_data[i]
                time_stamp = sim_time[i]

                imu = np.array([accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z], dtype=np.float32)
                eff_pred, t_inf = process_imu_and_predict(time_stamp, imu)

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

# %% Plot trace of covariance matrix

# plt.plot(vio.trace_covariance)
# plt.title('Trace of covariance matrix')

# %% Plot results

# fig = plt.figure()
#
# plt.subplot(121)
# plt.plot(euler[:, 0], label='yaw')
# plt.plot(euler[:, 1], label='pitch')
# plt.plot(euler[:, 2], label='roll')
# plt.ylabel('degrees')
# plt.title('Attitude of Quad')
# plt.legend()

# plt.subplot(122)
# plt.plot(translation[:, 0], label='Tx')
# plt.plot(translation[:, 1], label='Ty')
# plt.plot(translation[:, 2], label='Tz')
# plt.ylabel('meters')
# plt.title('Position of Quad')
# plt.legend()

#%%

# plt.figure()
# plt.plot(velocity[:, 0], label='vx')
# plt.plot(velocity[:, 1], label='vy')
# plt.plot(velocity[:, 2], label='vz')
# plt.ylabel('meters per second')
# plt.title('Velocity of Quad')
# plt.legend()

#%%
# plt.figure()
# plt.plot(a_bias[:, 0], label='ax')
# plt.plot(a_bias[:, 1], label='ay')
# plt.plot(a_bias[:, 2], label='az')
# plt.ylabel('meters per second squared')
# plt.title('Accelerometer Bias')
# plt.legend()

###############PLANNING PLOTTING##############################

# plot state vs vio_state
# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.
collision_pts = world.path_collisions(state['x'], robot_radius)


# increase the goal reached tolerance for VIO noisy state estimation and accumulated drift
goal_tolerance = 2
stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= goal_tolerance

no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

# print()
# print(f"Results:")
# print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
# print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
# print(f"  Flight time:     {flight_time:.1f} seconds")
# print(f"  Flight distance: {flight_distance:.1f} meters")
# print(f"  Planning time:   {planning_time:.1f} seconds")
# if not no_collision:
#     print()
#     print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# Visualize the original dense path from A*, your sparse waypoints, and the
# smooth trajectory.
# fig = plt.figure('A* Path, Waypoints, and Trajectory')
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# if hasattr(my_world_traj, 'path'):
#     if my_world_traj.path is not None:
#         world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
# else:
#     print("Have you set \'self.path\' in WorldTraj.__init__?")
# if hasattr(my_world_traj, 'points'):
#     if my_world_traj.points is not None:
#         world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
# else:
#     print("Have you set \'self.points\' in WorldTraj.__init__?")
# world.draw_line(ax, flat['x'], color='black', linewidth=2)
# ax.legend(handles=[
#     Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
#     Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
#     Line2D([], [], color='black', linewidth=2, label='Trajectory')],
#     loc='upper right')

# Position and Velocity vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
# x = state['x']
# x_des = flat['x']
# ax = axes[0]
# ax.plot(sim_time, x[:,0], 'r',    sim_time, x[:,1], 'g',    sim_time, x[:,2], 'b', linewidth=1, alpha=0.6)
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.plot(sim_time, x_des[:,0], 'k', sim_time, x_des[:,1], 'k', sim_time, x_des[:,2], 'k', linewidth=0.5, alpha=0.4)
# ax.set_ylabel('position, m')
# ax.grid('major')
# ax.set_title('Position')

# x_est = est_state['x']
# ax = axes[0]
# ax.plot(sim_time, x_des[:,0], 'r--', sim_time, x_des[:,1], 'g--', sim_time, x_des[:,2], 'b-', linewidth=2.5)
# ax.legend(('xest', 'yest', 'zest'), loc='lower right')

# v = state['v']
# v_des = flat['x_dot']
# ax = axes[1]
# ax.plot(sim_time, v[:,0], 'r',    sim_time, v[:,1], 'g',    sim_time, v[:,2], 'b', linewidth=1,alpha=0.6)
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.plot(sim_time, v_des[:,0], 'k', sim_time, v_des[:,1], 'k', sim_time, v_des[:,2], 'k', linewidth=0.5, alpha=0.5)
# ax.set_ylabel('velocity, m/s')
# ax.set_xlabel('time, s')
# ax.grid('major')

# v_est = est_state['v']
# ax = axes[1]
# ax.plot(sim_time, v_est[:,0], 'r--', sim_time, v_est[:,1], 'g--', sim_time, v_est[:,2], 'b--', linewidth=2.5)
# ax.legend(('xest', 'yest', 'zest'), loc='lower right')

# Orientation and Angular Velocity vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
# q_des = control['cmd_q']
# q = state['q']
# ax = axes[0]
# ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.', linewidth=1,alpha=0.6)
# ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
# ax.plot(sim_time, q_des[:,0], 'k', sim_time, q_des[:,1], 'k', sim_time, q_des[:,2], 'k', sim_time, q_des[:,3], 'k', linewidth=0.5, alpha=0.5)
# ax.set_ylabel('quaternion')
# ax.set_xlabel('time, s')
# ax.grid('major')

# q_est = est_state['q']
# ax = axes[0]
# ax.plot(sim_time, q_est[:,0], 'r.',    sim_time, q_est[:,1], 'g.',    sim_time, q_est[:,2], 'b.',    sim_time, q_est[:,3],     'k.', linewidth=2.5)
# ax.legend(('xest', 'yest', 'zest'), loc='lower right')

# w = state['w']
# ax = axes[1]
# ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('angular velocity, rad/s')
# ax.set_xlabel('time, s')
# ax.grid('major')

# w_est = est_state['w']
# ax = axes[1]
# ax.plot(sim_time, w_est[:,0], 'r--', sim_time, w_est[:,1], 'g--', sim_time, w_est[:,2], 'b--', linewidth=2.5)
# ax.legend(('xest', 'yest', 'zest'), loc='lower right')

# Commands vs. Time
# (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
# s = control['cmd_motor_speeds']
# ax = axes[0]
# ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
# ax.legend(('1', '2', '3', '4'), loc='upper right')
# ax.set_ylabel('motor speeds, rad/s')
# ax.grid('major')
# ax.set_title('Commands')
# M = control['cmd_moment']
# ax = axes[1]
# ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('moment, N*m')
# ax.grid('major')
# T = control['cmd_thrust']
# ax = axes[2]
# ax.plot(sim_time, T, 'k.')
# ax.set_ylabel('thrust, N')
# ax.set_xlabel('time, s')
# ax.grid('major')

# 3D Paths
# fig = plt.figure('3D Path')
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# world.draw_line(ax, flat['x'], color='black', linewidth=2)
# world.draw_points(ax, state['x'], color='blue', markersize=4)
# if collision_pts.size > 0:
#     ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
# ax.legend(handles=[
#     Line2D([], [], color='black', linewidth=2, label='Trajectory'),
#     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
#     loc='upper right')


# accelerometer_measurements = []
# for accel, _ in imu_measurements:
#     accelerometer_measurements.append(accel)
# accelerometer_measurements = np.array(accelerometer_measurements)
# plt.figure()
# plt.plot(sim_time[1:], accelerometer_measurements[: ,0], label='x')
# plt.plot(sim_time[1:], accelerometer_measurements[: ,1], label='y')
# plt.plot(sim_time[1:], accelerometer_measurements[: ,2], label='z')
# plt.title('Accelerometer Measurements')
# plt.legend()
# Animation (Slow)
#
# Instead of viewing the animation live, you may provide a .mp4 filename to save.

# R = Rotation.from_quat(state['q']).as_matrix()
# ani = animate(sim_time, state['x'], R, world=world, filename=None)

# plt.show()
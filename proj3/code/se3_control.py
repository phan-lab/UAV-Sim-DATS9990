import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.rotor_eff = np.ones(4)
        self.baseline_eff = np.ones(4)
        self.baseline_count = 0
        self.BASELINE_TIME = 2.0
        self.unrecoverable_counter = 0
        self.degraded_counter = 0
        self.eff_history = np.zeros((50, 4))
        self.hist_idx = 0
        self.mode = "NOMINAL"
        self.degraded_start_time = None

        # k_p_z = 7.5
        # k_d_z = 2 * np.sqrt(k_p_z) - 0.25
        #
        # k_R_yaw = 15
        # k_w_yaw = 2 * np.sqrt(k_R_yaw) - 0.75
        #
        # k_p_xy = 4.5
        # k_d_xy = 2 * np.sqrt(k_p_xy) + 2
        #
        # k_R_rp = 250
        # k_w_rp = 2 * np.sqrt(k_R_rp) - 7.5


        k_p_z = 7.5
        k_d_z = 2 * np.sqrt(k_p_z)

        k_R_yaw = 150
        k_w_yaw = 2 * np.sqrt(k_R_yaw)

        k_p_xy = 4.5
        k_d_xy = 2 * np.sqrt(k_p_xy)

        k_R_rp = 250
        k_w_rp = 2 * np.sqrt(k_R_rp)

        self.K_d = np.diag([k_d_xy, k_d_xy, k_d_z])
        self.K_p = np.diag([k_p_xy, k_p_xy, k_p_z])
        self.K_R = np.diag([k_R_rp, k_R_rp, k_R_yaw])
        self.K_w = np.diag([k_w_rp, k_w_rp, k_w_yaw])

        self.gamma = self.k_drag / self.k_thrust
        self.u_F = np.array([
            [1,1,1,1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [self.gamma, -self.gamma, self.gamma, -self.gamma]
        ])


    def set_rotor_effectiveness(self, eta):
        self.rotor_eff = 0.9 * self.rotor_eff + 0.1 * eta
        self.eff_history[self.hist_idx % 50] = self.rotor_eff
        self.hist_idx += 1


    def update_baseline(self, t):
        if t < self.BASELINE_TIME:
            self.baseline_eff = (
                self.baseline_eff * self.baseline_count + self.rotor_eff
            ) / (self.baseline_count + 1)
            self.baseline_count += 1


    def detect_mode(self, t):
        residual = self.rotor_eff / (self.baseline_eff + 1e-6)
        min_residual = np.min(residual)
        asym = np.max(residual) - np.min(residual)

        degraded = (
            t > self.BASELINE_TIME and
            min_residual < 0.8 and
            asym > 0.2
        )

        unrecoverable = (
            t > self.BASELINE_TIME and
            min_residual < 0.5
        )

        return degraded, unrecoverable


    def update_counters(self, degraded, unrecoverable):
        if degraded:
            self.degraded_counter += 1
        else:
            self.degraded_counter = max(self.degraded_counter - 1, 0)

        if unrecoverable:
            self.unrecoverable_counter += 1
        else:
            self.unrecoverable_counter = max(self.unrecoverable_counter - 1, 0)


    def update_mode(self, t):
        if self.mode == "NOMINAL":
            if self.unrecoverable_counter >= 20:
                self.mode = "UNRECOVERABLE"
            elif self.degraded_counter >= 20:
                self.mode = "DEGRADED"
                if self.degraded_start_time is None:
                    self.degraded_start_time = t
        elif self.mode == "DEGRADED":
            if self.unrecoverable_counter >= 20:
                self.mode = "UNRECOVERABLE"
            # elif self.degraded_counter == 0:
            #     self.mode = "NOMINAL"


    def reset(self):
        self.rotor_eff = np.ones(4)
        self.baseline_eff = np.ones(4)
        self.baseline_count = 0
        self.unrecoverable_counter = 0
        self.degraded_counter = 0
        self.eff_history = np.zeros((50, 4))
        self.hist_idx = 0
        self.mode = "NOMINAL"
        self.degraded_start_time = None


    def degraded_control(self, t, state):
        r = state["x"]
        r_dot = state["v"]
        q = state["q"]
        w = state["w"]

        if r[2] < 0.1:
            raise RuntimeError("DEGRADED_SAFE_LANDING")

        R = Rotation.from_quat(q).as_matrix()
        b3 = R[:, 2]

        t_degraded = t - self.degraded_start_time
        z_ref = max(0.0, r[2] - 0.3 * max(0.0, t_degraded - 1.0))

        z_err = r[2] - z_ref
        z_dot_err = r_dot[2]

        u1 = self.mass * (
            self.g
            - self.K_p[2, 2] * z_err
            - self.K_d[2, 2] * z_dot_err
        )
        u1 = np.clip(
            u1,
            0.6 * self.mass * self.g,
            1.1 * self.mass * self.g
        )

        R_des = np.eye(3)
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([
            e_R_mat[2, 1],
            e_R_mat[0, 2],
            e_R_mat[1, 0]
        ])

        e_w = w
        tau = self.inertia @ (
            - self.K_R @ e_R
            - self.K_w @ e_w
        )
        tau[2] = 0.0

        u = np.array([u1, tau[0], tau[1], 0.0])

        E_eff = self.u_F @ np.diag(self.rotor_eff)
        F = np.linalg.pinv(E_eff) @ u
        F = np.clip(F, 0.0, None)

        cmd_motor_speeds = np.sqrt(F / self.k_thrust)
        cmd_motor_speeds = np.clip(
            cmd_motor_speeds,
            self.rotor_speed_min,
            self.rotor_speed_max
        )

        return cmd_motor_speeds


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        r = state["x"]
        r_dot = state["v"]
        quaternion = state["q"]
        w = state["w"]

        r_T = flat_output["x"]
        r_dot_T = flat_output["x_dot"]
        r_ddot_T = flat_output["x_ddot"]
        r_dddot_T = flat_output["x_dddot"]
        r_ddddot_T = flat_output["x_ddddot"]
        yaw_T = flat_output["yaw"]
        yaw_dot_T = flat_output["yaw_dot"]

        R = Rotation.from_quat(quaternion).as_matrix()
        b_3 = R[:, 2]

        # print(f"Timestamp {t:.3f}, Predictions: {self.rotor_eff}")

        self.update_baseline(t)

        degraded, unrecoverable = self.detect_mode(t)
        self.update_counters(degraded, unrecoverable)
        self.update_mode(t)

        # UAV cannot be recovered, initiate emergency landing
        if self.mode == "UNRECOVERABLE":
            raise RuntimeError("EMERGENCY_LANDING")

        if self.mode == "DEGRADED":
            cmd_motor_speeds = self.degraded_control(t, state)
            return {
                'cmd_motor_speeds': cmd_motor_speeds,
                'cmd_thrust': 0,
                'cmd_moment': np.zeros(3),
                'cmd_q': np.zeros(4)
            }, self.mode

        r_ddot_des = r_ddot_T - self.K_d @ (r_dot - r_dot_T) - self.K_p @ (r - r_T)
        F_des = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g])
        u_1 = b_3 @ F_des

        b_3_des = F_des / np.linalg.norm(F_des)
        a_yaw = np.array([np.cos(yaw_T), np.sin(yaw_T), 0])
        b_2_des = np.cross(b_3_des, a_yaw) / np.linalg.norm(np.cross(b_3_des, a_yaw))
        b_1_des = np.cross(b_2_des, b_3_des)
        R_des = np.column_stack((b_1_des, b_2_des, b_3_des))

        e_R = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R[2, 1], e_R[0, 2], e_R[1, 0]])

        e_w = w
        u_2 = self.inertia @ (- self.K_R @ e_R - self.K_w @ e_w)

        u_vector = np.insert(u_2, 0, u_1)
        F_vector = np.linalg.solve(self.u_F, u_vector)

        F_vector = F_vector.clip(min=0)
        cmd_motor_speeds = np.sqrt(F_vector / self.k_thrust)

        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input, self.mode
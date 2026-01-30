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
        self.fault_mode = False
        self.fault_counter = 0

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


    def reset(self):
        self.rotor_eff = np.ones(4)
        self.fault_mode = False
        self.fault_counter = 0


    def safe_hover(self, state, flat_output):
        r = state["x"]
        r_dot = state["v"]
        r_T = flat_output["x"]
        r_dot_T = flat_output["x_dot"]
        z_err = r[2] - r_T[2]
        z_dot_err = r_dot[2] - r_dot_T[2]
        u_1 = self.mass * (
                self.g
                - self.K_p[2, 2] * z_err
                - self.K_d[2, 2] * z_dot_err
        )
        u_1 = np.clip(u_1, 0.5 * self.mass * self.g, 1.2 * self.mass * self.g)
        total_eff = np.sum(self.rotor_eff)
        total_eff = max(total_eff, 1e-3)
        omega_hover = np.sqrt(
            (u_1 / total_eff) / self.k_thrust
        )
        cmd_motor_speeds = np.ones(4) * omega_hover
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)
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

        if np.min(self.rotor_eff) < 0.3:
            cmd_motor_speeds = self.safe_hover(state, flat_output)
            return {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': 0,
                         'cmd_moment': np.zeros(3),
                         'cmd_q': np.zeros(4)}

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
        return control_input
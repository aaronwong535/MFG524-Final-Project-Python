import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
# import time  # Uncomment if needed for timing operations
# import dynamixel_sdk as dxl  # Uncomment if controlling real Dynamixel motors

# Helper functions for transformations
def rotx(theta):
    """Rotation matrix around x-axis"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def roty(theta):
    """Rotation matrix around y-axis"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotz(theta):
    """Rotation matrix around z-axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def trvec2tform(translation):
    """Convert translation vector to homogeneous transformation matrix"""
    T = np.eye(4)
    T[:3, 3] = translation
    return T

def rotm2tform(R):
    """Convert rotation matrix to homogeneous transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = R
    return T

# Robot class to simulate the 3-DOF manipulator
class Robot:
    def __init__(self):
        # Link lengths
        self.a1 = 0.055  # Length of link 1
        self.a2 = 0.0675  # Length of link 2
        self.a3 = 0.04  # Length of link 3
        
        # Masses
        self.m1 = 1.0  # Mass of link 1
        self.m2 = 1.0  # Mass of link 2
        self.m3 = 1.0  # Mass of link 3
        
        # Center of mass positions in local frames
        self.com1 = np.array([0, 0, self.a1/2])
        self.com2 = np.array([self.a2/2, 0, 0])
        self.com3 = np.array([self.a3/2, 0, 0])
        
        # Inertia tensors (simplified as in MATLAB code)
        self.I1 = np.diag([0.01, 0.01, 0.01])
        self.I2 = np.diag([0.01, 0.01, 0.01])
        self.I3 = np.diag([0.01, 0.01, 0.01])
        
        # Gravity vector
        self.gravity = np.array([0, 0, -9.81])
    
    def get_transform(self, q, target_frame='endEffector', base_frame='base'):
        """
        Calculate the transformation matrix from base_frame to target_frame
        Similar to MATLAB's getTransform function
        """
        q1, q2, q3 = q
        
        # Transformation matrices for each joint
        T01 = trvec2tform([0, 0, 0]) @ rotm2tform(rotz(q1))
        T12 = trvec2tform([0, 0, self.a1]) @ rotm2tform(np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])) @ rotm2tform(rotz(q2))
        T23 = trvec2tform([self.a2, 0, 0]) @ rotm2tform(rotz(q3))
        T3e = trvec2tform([self.a3, 0, 0]) @ rotm2tform(np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]))
        
        # Calculate transformations between different frames
        if base_frame == 'base' and target_frame == 'link1':
            return T01
        elif base_frame == 'base' and target_frame == 'link2':
            return T01 @ T12
        elif base_frame == 'base' and target_frame == 'link3':
            return T01 @ T12 @ T23
        elif base_frame == 'base' and target_frame == 'endEffector':
            return T01 @ T12 @ T23 @ T3e
        else:
            # Default to end-effector if frames not recognized
            return T01 @ T12 @ T23 @ T3e
    
    def mass_matrix(self, q):
        """
        Calculate the mass matrix for the robot
        Simplified implementation that approximates MATLAB's massMatrix function
        """
        # For a simple implementation, we'll use a constant mass matrix
        # In reality, the mass matrix would depend on the robot's configuration
        M = np.diag([0.5, 0.3, 0.1])
        return M
    
    def velocity_product(self, q, q_dot):
        """
        Calculate Coriolis and centrifugal terms
        Simplified implementation that approximates MATLAB's velocityProduct function
        """
        # For a simple implementation, we'll use zeros
        # In reality, these terms would be complex nonlinear functions
        return np.zeros(3)
    
    def gravity_torque(self, q):
        """
        Calculate gravity torque vector
        Simplified implementation that approximates MATLAB's gravityTorque function
        """
        # Simple gravity effect model
        G = np.zeros(3)
        q1, q2, q3 = q
        
        # No gravity effect on first joint (rotation around gravity axis)
        G[0] = 0
        
        # Effect on second joint
        G[1] = self.m2 * 9.81 * self.a2/2 * np.cos(q2 + np.pi/2)
        
        # Effect on third joint
        G[2] = self.m3 * 9.81 * self.a3/2 * np.cos(q2 + q3 + np.pi/2)
        
        return G

# Main simulation function
def main():
    # Create robot model
    robot = Robot()
    
    # Define PID controller parameters
    Kp = np.array([40, 40, 20])  # Proportional gains
    Ki = np.array([1, 0.5, 0.2])  # Integral gains
    Kd = np.array([10, 2, 2])  # Derivative gains
    
    # Initial joint angles and velocities
    q0 = np.zeros(3)
    q_dot0 = np.zeros(3)
    
    # Desired joint angles (target pose)
    q_desired = np.zeros(3)
    
    # Simulation parameters
    t_start = 0
    t_end = 3
    dt = 0.01
    tspan = np.arange(t_start, t_end + dt, dt)
    num_steps = len(tspan)
    log_dt = 0.1  # 10 Hz = 0.1 seconds
    log_steps = int(t_end / log_dt) + 1
    t_log = np.arange(t_start, t_end + log_dt, log_dt)
    
    # Initialize variables for simulation
    q_history = np.zeros((num_steps, 3))
    q_dot_history = np.zeros((num_steps, 3))
    tau_history = np.zeros((num_steps, 3))
    error_history = np.zeros((num_steps, 3))
    q_log_history = np.zeros((log_steps, 3))
    
    q = q0.copy()
    q_dot = q_dot0.copy()
    q_history[0] = q
    q_dot_history[0] = q_dot
    q_log_history[0] = q
    
    # Initialize error integral for PID controller
    error_integral = np.zeros(3)
    
    # Manipulator dynamics function for integration
    def manipulator_dynamics(t, state, tau):
        """
        Dynamics function for the manipulator
        t: time
        state: [q1, q2, q3, q1_dot, q2_dot, q3_dot]
        tau: control torques
        """
        q = state[:3]
        q_dot = state[3:]
        
        M = robot.mass_matrix(q)
        C = robot.velocity_product(q, q_dot)
        G = robot.gravity_torque(q)
        
        # Calculate joint accelerations using inverse dynamics
        q_ddot = np.linalg.solve(M, tau - C - G)
        
        return np.concatenate([q_dot, q_ddot])
    
    # Main simulation loop
    log_counter = 0
    next_log_time = t_start + log_dt
    
    for i in range(1, num_steps):
        current_time = tspan[i]
        dt = tspan[i] - tspan[i-1]
        
        # Calculate error terms
        error = q_desired - q
        error_integral += error * dt
        error_derivative = -q_dot
        
        # PID Control Law
        tau = Kp * error + Ki * error_integral + Kd * error_derivative
        tau_max = 20
        tau = np.clip(tau, -tau_max, tau_max)
        
        error_history[i] = error
        
        # Simulate dynamics using scipy's ODE solver
        sol = solve_ivp(
            lambda t, y: manipulator_dynamics(t, y, tau),
            [tspan[i-1], tspan[i]],
            np.concatenate([q, q_dot]),
            method='RK45',
            t_eval=[tspan[i]]
        )
        
        state_temp = sol.y
        q = state_temp[:3, -1]
        q_dot = state_temp[3:, -1]
        
        q_history[i] = q
        q_dot_history[i] = q_dot
        tau_history[i] = tau
        
        # Log at 10 Hz
        if current_time >= next_log_time and log_counter < log_steps - 1:
            log_counter += 1
            q_log_history[log_counter] = q
            next_log_time += log_dt
    
    # Calculate end-effector positions over time
    ee_positions = np.zeros((num_steps, 3))
    ee_positions_log = np.zeros((log_steps, 3))
    
    for i in range(num_steps):
        T = robot.get_transform(q_history[i])
        ee_positions[i] = T[:3, 3]
    
    for i in range(log_steps):
        T = robot.get_transform(q_log_history[i])
        ee_positions_log[i] = T[:3, 3]
    
    # Plot results
    # Joint Positions Plot
    plt.figure(figsize=(10, 8))
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, q_history[:, j], 'b-', label='Simulated')
        plt.plot(t_log[:log_counter+1], q_log_history[:log_counter+1, j], 'g-o', label='Logged (10 Hz)')
        plt.plot(tspan, np.ones_like(tspan) * q_desired[j], 'r--', label='Desired')
        plt.ylabel(f'Joint {j+1} (rad)')
        plt.legend()
        if j == 0:
            plt.title('Joint Positions')
        if j == 2:
            plt.xlabel('Time (s)')
    
    # Enhanced End-Effector Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full trajectory
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=1.5, label='Full Trajectory')
    
    # Plot 10 Hz logged points
    ax.plot(ee_positions_log[:log_counter+1, 0], ee_positions_log[:log_counter+1, 1], 
            ee_positions_log[:log_counter+1, 2], 'k*', markersize=6, label='10 Hz Points')
    
    # Plot start and end points
    ax.plot([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]], 'go', markersize=10, linewidth=2, label='Start')
    ax.plot([ee_positions[-1, 0]], [ee_positions[-1, 1]], [ee_positions[-1, 2]], 'ro', markersize=10, linewidth=2, label='End')
    
    ax.grid(True)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('End-Effector Trajectory', fontsize=14)
    ax.legend(loc='best')
    ax.view_init(-37.5, 30)  # Adjust view angle for better visualization
    ax.set_box_aspect([1, 1, 1])  # Equivalent to axis equal in MATLAB
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from dynamixel_sdk import *  # Import dynamixel SDK

class RigidBody:
    def __init__(self, name, mass=0, center_of_mass=None, inertia=None):
        self.name = name
        self.mass = mass
        self.center_of_mass = np.zeros(3) if center_of_mass is None else np.array(center_of_mass)
        self.inertia = np.zeros(6) if inertia is None else np.array(inertia)
        self.joint = None
        self.parent = None
        self.children = []
        self.transform = np.eye(4)  # Identity transformation matrix

class Joint:
    def __init__(self, name, joint_type, joint_axis=None, transform=None):
        self.name = name
        self.joint_type = joint_type  # 'revolute' or 'fixed'
        self.joint_axis = np.array([0, 0, 1]) if joint_axis is None else np.array(joint_axis)
        self.transform = np.eye(4) if transform is None else transform

class RigidBodyTree:
    def __init__(self, gravity=None):
        self.bodies = {}
        self.base = RigidBody("base")
        self.bodies["base"] = self.base
        self.gravity = np.array([0, 0, -9.81]) if gravity is None else np.array(gravity)
    
    def add_body(self, body, parent_name):
        if parent_name in self.bodies:
            parent = self.bodies[parent_name]
            body.parent = parent
            parent.children.append(body)
            self.bodies[body.name] = body
        else:
            raise ValueError(f"Parent body '{parent_name}' not found in the tree")

def trvec2tform(translation):
    """Convert translation vector to homogeneous transformation matrix"""
    tform = np.eye(4)
    tform[:3, 3] = translation
    return tform

def rotm2tform(rotation_matrix):
    """Convert rotation matrix to homogeneous transformation matrix"""
    tform = np.eye(4)
    tform[:3, :3] = rotation_matrix
    return tform

def mass_matrix(robot, q):
    """Calculate the mass matrix at the given joint configuration"""
    # This is a simplified implementation
    n = len(q)
    M = np.eye(n)
    
    # For each link, add its contribution to the mass matrix
    for i, body_name in enumerate(['link1', 'link2', 'link3']):
        if body_name in robot.bodies:
            body = robot.bodies[body_name]
            M[i, i] = body.mass  # Simplified inertia model
    
    return M

def velocity_product(robot, q, q_dot):
    """Calculate the Coriolis and centrifugal terms"""
    # Simplified implementation
    n = len(q)
    C = np.zeros(n)
    
    # Simple Coriolis term proportional to velocity squared
    for i in range(n):
        C[i] = 0.1 * q_dot[i]**2  # Simple model
    
    return C

def gravity_torque(robot, q):
    """Calculate the gravity torque at the given joint configuration"""
    # Simplified implementation
    n = len(q)
    G = np.zeros(n)
    
    # For each link, calculate the gravity effect
    for i, body_name in enumerate(['link1', 'link2', 'link3']):
        if body_name in robot.bodies:
            body = robot.bodies[body_name]
            # Simplified gravity model
            if i == 0:  # First joint mainly affected by link1
                G[i] = body.mass * 9.81 * 0.055 * np.sin(q[i])
            elif i == 1:  # Second joint mainly affected by link2
                G[i] = body.mass * 9.81 * 0.0675 * np.sin(q[i])
            elif i == 2:  # Third joint mainly affected by link3
                G[i] = body.mass * 9.81 * 0.04 * np.sin(q[i])
    
    return G

def geometric_jacobian(robot, q, end_effector_name):
    """Calculate the geometric Jacobian at the given joint configuration"""
    # Simplified implementation
    J = np.zeros((6, len(q)))
    
    # For a 3-DOF planar robot, simplified Jacobian
    # This is a highly simplified model
    J[0, 0] = -0.0675 * np.sin(q[0]) - 0.04 * np.sin(q[0] + q[1])
    J[0, 1] = -0.04 * np.sin(q[0] + q[1])
    J[0, 2] = 0
    
    J[1, 0] = 0.0675 * np.cos(q[0]) + 0.04 * np.cos(q[0] + q[1])
    J[1, 1] = 0.04 * np.cos(q[0] + q[1])
    J[1, 2] = 0
    
    # Rotational part
    J[3:6, 0] = np.array([0, 0, 1])
    J[3:6, 1] = np.array([0, 0, 1])
    J[3:6, 2] = np.array([0, 0, 1])
    
    return J

def get_transform(robot, q, end_effector_name, base_name):
    """Calculate the transformation matrix from base to end-effector"""
    # Simple forward kinematics for a 3-DOF robot
    a1 = 0.055
    a2 = 0.0675
    a3 = 0.04
    
    T = np.eye(4)
    
    # Joint 1 transformation
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    T1 = np.array([
        [c1, -s1, 0, 0],
        [s1, c1, 0, 0],
        [0, 0, 1, a1],
        [0, 0, 0, 1]
    ])
    T = T @ T1
    
    # Joint 2 transformation
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    T2 = np.array([
        [c2, -s2, 0, a2],
        [s2, c2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T = T @ T2
    
    # Joint 3 transformation
    c3, s3 = np.cos(q[2]), np.sin(q[2])
    T3 = np.array([
        [c3, -s3, 0, a3],
        [s3, c3, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T = T @ T3
    
    return T

def manipulator_dynamics(t, state, robot, tau):
    """Define the system's equations of motion"""
    q = state[:3]
    q_dot = state[3:]
    
    M = mass_matrix(robot, q)
    C = velocity_product(robot, q, q_dot)
    G = gravity_torque(robot, q)
    
    # Calculate acceleration
    M_inv = np.linalg.inv(M)
    q_ddot = M_inv @ (tau - C - G)
    
    return np.concatenate([q_dot, q_ddot])

def rk4_step(f, t, y, h, *args):
    """Fourth-order Runge-Kutta integrator"""
    k1 = h * np.array(f(t, y, *args))
    k2 = h * np.array(f(t + 0.5 * h, y + 0.5 * k1, *args))
    k3 = h * np.array(f(t + 0.5 * h, y + 0.5 * k2, *args))
    k4 = h * np.array(f(t + h, y + k3, *args))
    
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def main():
    # 1. Define Robot Model
    robot = RigidBodyTree()
    
    # Link lengths
    a1 = 0.055
    a2 = 0.0675
    a3 = 0.04
    
    # Create link 1
    link1 = RigidBody('link1', mass=0.2, center_of_mass=[0, 0, a1/2], inertia=[0.01, 0.01, 0.01, 0, 0, 0])
    joint1 = Joint('joint1', 'revolute', [0, 0, 1])
    joint1.transform = trvec2tform([0, 0, 0])
    link1.joint = joint1
    robot.add_body(link1, 'base')
    
    # Create link 2
    link2 = RigidBody('link2', mass=0.2, center_of_mass=[a2/2, 0, 0], inertia=[0.01, 0.01, 0.01, 0, 0, 0])
    joint2 = Joint('joint2', 'revolute', [0, 0, 1])
    rot_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    joint2.transform = trvec2tform([0, 0, a1]) @ rotm2tform(rot_matrix)
    link2.joint = joint2
    robot.add_body(link2, 'link1')
    
    # Create link 3
    link3 = RigidBody('link3', mass=0.2, center_of_mass=[a3/2, 0, 0], inertia=[0.01, 0.01, 0.01, 0, 0, 0])
    joint3 = Joint('joint3', 'revolute', [0, 0, 1])
    joint3.transform = trvec2tform([a2, 0, 0])
    link3.joint = joint3
    robot.add_body(link3, 'link2')
    
    # Create end-effector frame
    end_effector = RigidBody('endEffector')
    end_effector_joint = Joint('endEffectorJoint', 'fixed')
    rot_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    end_effector_joint.transform = trvec2tform([a3, 0, 0]) @ rotm2tform(rot_matrix)
    end_effector.joint = end_effector_joint
    robot.add_body(end_effector, 'link3')
    
    robot.gravity = np.array([0, 0, -9.81])
    
    # 2. Define Controller Parameters
    Kp = np.array([20, 15, 10])  # Proportional gains
    Ki = np.array([1, 0.5, 0.2])  # Integral gains
    Kd = np.array([5, 2, 1])     # Derivative gains
    
    # 3. Momentum Observer Parameters
    K_obs = np.diag([2, 2, 2])  # Observer gain matrix
    
    # Initial conditions
    q0 = np.array([0, 0, 0])       # Initial joint angles
    q_dot0 = np.array([0, 0, 0])   # Initial joint velocities
    p_hat = np.zeros(3)            # Initial estimated momentum
    tau_ext_hat = np.zeros(3)      # Initial estimated external torque
    integral_term = np.zeros(3)    # Integral term for the formula
    
    # 4. Simulation Parameters
    t_start = 0
    t_end = 5
    dt = 0.01
    tspan = np.arange(t_start, t_end + dt, dt)
    num_steps = len(tspan)
    
    # Target position (holding position)
    q_desired = np.array([0, 0, 0])
    
    # 5. Define External Force Profile
    mass = 0.1
    g = 9.81
    F_ext_magnitude = mass * g  # N
    F_ext_direction = np.array([0, 0, -1])  # Unit vector in negative z-direction
    F_ext_start = 1.5  # start time (seconds)
    F_ext_end = 3.5    # end time (seconds)
    
    # 6. Initialize Storage Variables
    q_history = np.zeros((num_steps, 3))
    q_dot_history = np.zeros((num_steps, 3))
    tau_history = np.zeros((num_steps, 3))
    tau_ext_history = np.zeros((num_steps, 3))
    p_hat_history = np.zeros((num_steps, 3))
    r_history = np.zeros((num_steps, 3))  # Residual
    error_history = np.zeros((num_steps, 3))
    F_ext_applied_history = np.zeros((num_steps, 3))
    
    q = q0.copy()
    q_dot = q_dot0.copy()
    q_history[0, :] = q
    q_dot_history[0, :] = q_dot
    
    # 7. Main Simulation Loop with Momentum Observer
    # Calculate initial mass matrix and momentum
    M_initial = mass_matrix(robot, q0)
    p = M_initial @ q_dot0  # Initialize momentum
    p_hat_history[0, :] = p
    
    error_integral = np.zeros(3)
    
    for i in range(1, num_steps):
        current_time = tspan[i]
        dt = tspan[i] - tspan[i-1]
        
        # Get robot dynamics at current state
        M = mass_matrix(robot, q)
        C = velocity_product(robot, q, q_dot)
        G = gravity_torque(robot, q)
        
        # Calculate control torque (PID)
        error = q_desired - q
        error_integral += error * dt
        error_derivative = -q_dot
        
        tau_pid = Kp * error + Ki * error_integral + Kd * error_derivative
        tau_ctrl = tau_pid + G
        error_history[i, :] = error
        
        # Compute current momentum
        p = M @ q_dot
        
        # Update the integral term according to the formula
        # integrand = u + C - G + ŵ_e
        integrand = tau_ctrl + C - G + tau_ext_hat
        integral_term += integrand * dt
        
        # Calculate the residual according to the formula
        # r = p - ∫(...)dt
        r = p - integral_term
        
        # Update the estimated external torque directly using the formula
        # ŵ_e = K_M * r
        tau_ext_hat = K_obs @ r
        
        # Store residual
        r_history[i, :] = r
        
        # Apply external force (if within the time window)
        F_ext = np.zeros(3)
        if F_ext_start <= current_time <= F_ext_end:
            F_ext = F_ext_magnitude * F_ext_direction
        
        # Convert external force to joint torques using the Jacobian
        J = geometric_jacobian(robot, q, 'endEffector')
        J_linear = J[3:6, :]  # Extract the linear part (for force)
        tau_ext = J_linear.T @ F_ext
        
        F_ext_applied_history[i, :] = F_ext
        
        # Combine control and external torques
        tau_total = tau_ctrl + tau_ext
        
        # Simulate dynamics using RK4 integrator
        state = np.concatenate([q, q_dot])
        state = rk4_step(manipulator_dynamics, tspan[i-1], state, dt, robot, tau_total)
        
        q = state[:3]
        q_dot = state[3:]
        
        # Store history
        q_history[i, :] = q
        q_dot_history[i, :] = q_dot
        tau_history[i, :] = tau_ctrl
        tau_ext_history[i, :] = tau_ext_hat
        p_hat_history[i, :] = p
    
    # 8. Calculate End-Effector Positions Over Time
    ee_positions = np.zeros((num_steps, 3))
    for i in range(num_steps):
        T = get_transform(robot, q_history[i, :], 'endEffector', 'base')
        ee_positions[i, :] = T[:3, 3]
    
    # 9. Plot Results
    # Plot 1: Joint Positions
    plt.figure(figsize=(12, 9))
    plt.suptitle('Joint Positions')
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, q_history[:, j], 'b-', linewidth=1.5)
        plt.plot(tspan, np.ones_like(tspan) * q_desired[j], 'r--', linewidth=1)
        plt.ylabel(f'Joint {j+1} (rad)')
        if j == 0:
            plt.title('Joint Positions')
        if j == 2:
            plt.xlabel('Time (s)')
        plt.grid(True)
    plt.tight_layout()
    
    # Plot 2: Estimated External Torques (main result)
    plt.figure(figsize=(12, 9))
    plt.suptitle('Estimated External Torques (Joint Space)')
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, tau_ext_history[:, j], 'b-', linewidth=1.5)
        
        # Plot the time window when external force is applied
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=F_ext_start, color='g', linestyle='--', label='Force Applied')
        plt.axvline(x=F_ext_end, color='r', linestyle='--', label='Force Removed')
        
        plt.ylabel(f'Joint {j+1} (Nm)')
        if j == 0:
            plt.title('Estimated External Torques (Wrench Observer)')
        if j == 2:
            plt.xlabel('Time (s)')
        plt.grid(True)
        if j == 0:
            plt.legend(['Estimated Torque', '', 'Force Applied', 'Force Removed'])
    plt.tight_layout()
    
    # Plot 3: External Force Applied
    plt.figure(figsize=(12, 9))
    plt.suptitle('Applied External Force')
    directions = ['X', 'Y', 'Z']
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, F_ext_applied_history[:, j], 'r-', linewidth=1.5)
        
        plt.ylabel(f'Force {directions[j]} (N)')
        if j == 0:
            plt.title('External Force Applied to End Effector')
        if j == 2:
            plt.xlabel('Time (s)')
        plt.grid(True)
    plt.tight_layout()
    
    # Plot 4: End-Effector Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=1.5)
    ax.plot([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]], 'go', markersize=10, linewidth=2)
    ax.plot([ee_positions[-1, 0]], [ee_positions[-1, 1]], [ee_positions[-1, 2]], 'ro', markersize=10, linewidth=2)
    
    # Mark the time when force is applied
    force_start_idx = np.where(tspan >= F_ext_start)[0][0]
    force_end_idx = np.where(tspan >= F_ext_end)[0][0]
    ax.plot([ee_positions[force_start_idx, 0]], [ee_positions[force_start_idx, 1]], 
            [ee_positions[force_start_idx, 2]], 'ms', markersize=10, linewidth=2)
    ax.plot([ee_positions[force_end_idx, 0]], [ee_positions[force_end_idx, 1]], 
            [ee_positions[force_end_idx, 2]], 'cs', markersize=10, linewidth=2)
    
    ax.grid(True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory')
    ax.legend(['Trajectory', 'Start', 'End', 'Force Applied', 'Force Removed'])
    ax.view_init(-37.5, 30)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Plot 5: Momentum Residual
    plt.figure(figsize=(12, 9))
    plt.suptitle('Momentum Residual')
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, r_history[:, j], 'b-', linewidth=1.5)
        
        plt.ylabel(f'Joint {j+1} Residual')
        if j == 0:
            plt.title('Momentum Observer Residual')
        if j == 2:
            plt.xlabel('Time (s)')
        plt.grid(True)
    plt.tight_layout()
    
    # Plot 6: Control Torques (tau_ctrl)
    plt.figure(figsize=(12, 9))
    plt.suptitle('Control Torques')
    for j in range(3):
        plt.subplot(3, 1, j+1)
        plt.plot(tspan, tau_history[:, j], 'b-', linewidth=1.5)
        
        # Plot the time window when external force is applied
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=F_ext_start, color='g', linestyle='--', label='Force Applied')
        plt.axvline(x=F_ext_end, color='r', linestyle='--', label='Force Removed')
        
        plt.ylabel(f'Joint {j+1} (Nm)')
        if j == 0:
            plt.title('Control Torques (PID Controller)')
        if j == 2:
            plt.xlabel('Time (s)')
        plt.grid(True)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()
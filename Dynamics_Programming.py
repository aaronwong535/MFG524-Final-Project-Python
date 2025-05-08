import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.integrate import solve_ivp

class RigidBodyTree:
    def __init__(self):
        # Initialize robot with empty links
        self.links = []
        self.gravity = np.array([0, 0, -9.81])
        
    def add_link(self, link, parent_name):
        # Add link to robot
        link.parent_name = parent_name
        self.links.append(link)
    
    def mass_matrix(self, q):
        # Compute mass matrix (simplified)
        n = len(q)
        M = np.eye(n)
        for i in range(n):
            M[i, i] = self.links[i].mass
        return M
    
    def velocity_product(self, q, q_dot):
        # Compute Coriolis and centrifugal terms (simplified)
        n = len(q)
        C = np.zeros(n)
        return C
    
    def gravity_torque(self, q):
        # Compute gravity torque
        n = len(q)
        G = np.zeros(n)
        
        # Transform matrices for each link
        T = self._get_transforms(q)
        
        for i in range(n):
            # Position of center of mass in world frame
            com_pos = T[i] @ np.append(self.links[i].center_of_mass, 1)
            
            # Force due to gravity
            force = self.links[i].mass * self.gravity
            
            # Torque due to gravity (simplified)
            G[i] = force[2] * com_pos[0]  # Simplified moment arm calculation
        
        return G
    
    def _get_transforms(self, q):
        # Compute transformation matrices for each link
        n = len(q)
        T = [np.eye(4)]  # Base transform
        
        for i in range(n):
            link = self.links[i]
            # Joint transform
            if link.joint_type == 'revolute':
                joint_transform = rotation_z(q[i])
            else:
                joint_transform = np.eye(4)
            
            # Link transform
            link_transform = link.transform
            
            # Combined transform
            if i == 0:
                T.append(T[0] @ joint_transform @ link_transform)
            else:
                T.append(T[i] @ joint_transform @ link_transform)
        
        return T
    
    def forward_kinematics(self, q):
        # Compute end-effector position
        T = self._get_transforms(q)
        return T[-1][:3, 3]  # Extract position from last transform
    
    def show(self, q):
        # Visualize robot
        T = self._get_transforms(q)
        
        # Extract positions
        positions = np.zeros((len(T), 3))
        for i in range(len(T)):
            positions[i] = T[i][:3, 3]
        
        # Plot robot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot links
        for i in range(1, len(positions)):
            ax.plot([positions[i-1][0], positions[i][0]],
                    [positions[i-1][1], positions[i][1]],
                    [positions[i-1][2], positions[i][2]], 'bo-', linewidth=2)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3-DOF Robot')
        
        plt.draw()
        plt.pause(0.001)
        
        return fig, ax

class RigidBody:
    def __init__(self, name, joint_type='fixed'):
        self.name = name
        self.joint_type = joint_type
        self.transform = np.eye(4)
        self.joint_axis = np.array([0, 0, 1])  # Default rotation around z-axis
        self.parent_name = None
        self.mass = 1.0
        self.center_of_mass = np.zeros(3)
        self.inertia = np.eye(3)
    
    def set_transform(self, transform):
        self.transform = transform

# Helper functions for transformations
def translation(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

def rotation_z(theta):
    T = np.eye(4)
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    return T

def rotation_matrix(axis, angle):
    # Convert to rotation matrix
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = axis * np.sin(angle / 2)
    
    R = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    return T

def manipulator_dynamics(t, state, robot, tau):
    n = len(tau)
    q = state[:n]
    q_dot = state[n:]
    
    M = robot.mass_matrix(q)
    C = robot.velocity_product(q, q_dot)
    G = robot.gravity_torque(q)
    
    q_ddot = np.linalg.solve(M, tau - C - G)
    
    return np.concatenate([q_dot, q_ddot])

def main():
    # Create robot model
    robot = RigidBodyTree()
    
    # Link lengths
    a1 = 0.055
    a2 = 0.0675
    a3 = 0.04
    
    # Create link 1
    body1 = RigidBody('link1', 'revolute')
    body1.set_transform(translation(0, 0, a1))
    body1.mass = 1
    body1.center_of_mass = np.array([0, 0, a1/2])
    body1.inertia = np.diag([0.01, 0.01, 0.01])
    robot.add_link(body1, 'base')
    
    # Create link 2
    body2 = RigidBody('link2', 'revolute')
    # Rotation matrix equivalent to [1 0 0; 0 0 -1; 0 1 0]
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    T = np.eye(4)
    T[:3, :3] = R
    body2.set_transform(translation(0, 0, a1) @ T)
    body2.mass = 1
    body2.center_of_mass = np.array([a2/2, 0, 0])
    body2.inertia = np.diag([0.01, 0.01, 0.01])
    robot.add_link(body2, 'link1')
    
    # Create link 3
    body3 = RigidBody('link3', 'revolute')
    body3.set_transform(translation(a2, 0, 0))
    body3.mass = 1
    body3.center_of_mass = np.array([a3/2, 0, 0])
    body3.inertia = np.diag([0.01, 0.01, 0.01])
    robot.add_link(body3, 'link2')
    
    # Create end-effector frame
    end_effector = RigidBody('endEffector', 'fixed')
    # Rotation matrix equivalent to [0 0 1; 1 0 0; 0 1 0]
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    T = np.eye(4)
    T[:3, :3] = R
    end_effector.set_transform(translation(a3, 0, 0) @ T)
    robot.add_link(end_effector, 'link3')
    
    # Define initial conditions
    q0 = np.array([0, 0, 0])  # Initial joint angles
    q_dot0 = np.array([0, 0, 0])  # Initial joint velocities
    
    # Define simulation parameters
    t_start = 0
    t_end = 0.1
    dt = 0.01
    t_span = (t_start, t_end)
    
    # Define control torques
    tau = np.array([0, 0, 0])  # Torques for each joint
    
    # Display gravity torque at initial position
    G_initial = robot.gravity_torque(q0)
    print("Gravity torque at initial position:")
    print(G_initial)
    
    # Simulation
    initial_state = np.concatenate([q0, q_dot0])
    
    # Define the ODE solver parameters
    t_eval = np.arange(t_start, t_end + dt, dt)
    
    # Solve the ODE
    solution = solve_ivp(
        lambda t, y: manipulator_dynamics(t, y, robot, tau),
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45'
    )
    
    # Extract results
    t = solution.t
    q = solution.y[:3, :].T
    q_dot = solution.y[3:, :].T
    
    # Visualize robot animation
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.ion()  # Turn on interactive mode
    
    for i in range(len(t)):
        plt.clf()  # Clear the figure
        robot.show(q[i])
        plt.pause(0.01)
    
    plt.ioff()  # Turn off interactive mode
    
    # Create joint position plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, q[:, 0], 'r-', linewidth=2, label='Joint 1')
    plt.plot(t, q[:, 1], 'g-', linewidth=1.5, label='Joint 2')
    plt.plot(t, q[:, 2], 'b-', linewidth=1.5, label='Joint 3')
    plt.grid(True)
    plt.title('Joint Positions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angles (rad)')
    plt.legend()
    plt.xlim([t_start, t_end])
    plt.axhline(y=0, color='k', linestyle='--', label='Zero Position')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
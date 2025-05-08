import numpy as np
import matplotlib.pyplot as plt
import time
from dynamixel_sdk import *  # Import Dynamixel SDK

# Configuration parameters
PORT = 'COM5'           # Serial port - update as needed
BAUDRATE = 1000000      # Baud rate for AX-12A
PROTOCOL_VERSION = 1.0  # Protocol version for AX-12A (1.0)
MOTOR_IDS = [1, 2, 3]   # Three motor IDs
NUM_MOTORS = len(MOTOR_IDS)
PUBLISH_FREQ = 15       # Hz
DURATION = 10           # Seconds
TOTAL_SAMPLES = PUBLISH_FREQ * DURATION

# Position parameters
HOME_POS = 512          # Center position (0-1023) for all servos

# Dynamixel Address Table for AX-12A
ADDR_AX_TORQUE_ENABLE       = 24
ADDR_AX_GOAL_POSITION       = 30
ADDR_AX_PRESENT_POSITION    = 36
ADDR_AX_PRESENT_SPEED       = 38
ADDR_AX_MOVING              = 46

# Protocol values
TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0
DXL_MOVING_STATUS_THRESHOLD = 10  # Threshold for detecting servo movement

# Robot Model Setup - Link lengths
a1 = 0.055
a2 = 0.0675
a3 = 0.04


class RobotModel:
    """Custom robot model class for 3-DOF manipulator"""
    
    def __init__(self, a1, a2, a3):
        """Initialize the robot model with link lengths"""
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        
        # Set masses of links
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        
        # Set center of mass positions
        self.com1 = np.array([0, 0, a1/2])
        self.com2 = np.array([a2/2, 0, 0])
        self.com3 = np.array([a3/2, 0, 0])
        
        # Set inertia matrices (simplified diagonal)
        self.I1 = np.diag([0.01, 0.01, 0.01])
        self.I2 = np.diag([0.01, 0.01, 0.01])
        self.I3 = np.diag([0.01, 0.01, 0.01])
        
        # Set gravity vector
        self.gravity = np.array([0, 0, -9.81])
    
    def forward_kinematics(self, q):
        """Compute forward kinematics for the robot"""
        q1, q2, q3 = q
        
        # Joint 1 transformation (base to joint 1)
        T01 = np.array([
            [np.cos(q1), -np.sin(q1), 0, 0],
            [np.sin(q1), np.cos(q1), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Joint 2 transformation (joint 1 to joint 2)
        T12 = np.array([
            [np.cos(q2), 0, np.sin(q2), 0],
            [0, 1, 0, 0],
            [-np.sin(q2), 0, np.cos(q2), self.a1],
            [0, 0, 0, 1]
        ])
        
        # Joint 3 transformation (joint 2 to joint 3)
        T23 = np.array([
            [np.cos(q3), -np.sin(q3), 0, self.a2],
            [np.sin(q3), np.cos(q3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # End effector transformation (joint 3 to end effector)
        T3e = np.array([
            [0, 0, 1, self.a3],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        
        # Compute transformation matrices
        T02 = T01 @ T12
        T03 = T02 @ T23
        T0e = T03 @ T3e
        
        return T01, T02, T03, T0e
    
    def jacobian(self, q):
        """Compute the geometric Jacobian at the end-effector"""
        T01, T02, T03, T0e = self.forward_kinematics(q)
        
        # Extract rotation matrices
        R01 = T01[:3, :3]
        R02 = T02[:3, :3]
        R03 = T03[:3, :3]
        
        # Extract positions
        p0e = T0e[:3, 3]
        p01 = T01[:3, 3]
        p02 = T02[:3, 3]
        p03 = T03[:3, 3]
        
        # Define joint axes in local frames
        z0 = np.array([0, 0, 1])  # Joint 1 axis in base frame
        z1 = R01 @ np.array([0, 0, 1])  # Joint 2 axis in base frame
        z2 = R02 @ np.array([0, 0, 1])  # Joint 3 axis in base frame
        
        # Compute Jacobian columns
        J1_v = np.cross(z0, p0e - np.array([0, 0, 0]))
        J2_v = np.cross(z1, p0e - p01)
        J3_v = np.cross(z2, p0e - p02)
        
        J1_w = z0
        J2_w = z1
        J3_w = z2
        
        # Combine into full Jacobian
        J = np.zeros((6, 3))
        J[:3, 0] = J1_v
        J[:3, 1] = J2_v
        J[:3, 2] = J3_v
        J[3:, 0] = J1_w
        J[3:, 1] = J2_w
        J[3:, 2] = J3_w
        
        return J
    
    def inertia(self, q):
        """Compute the mass matrix (inertia matrix)"""
        q1, q2, q3 = q
        
        # Mass matrix
        M = np.zeros((3, 3))
        
        # Add contribution from link 1
        M[0, 0] += self.m1 * (self.com1[0]**2 + self.com1[1]**2) + self.I1[2, 2]
        
        # Add contribution from link 2
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        M[0, 0] += self.m2 * ((self.a1*s2 + self.com2[0]*c2)**2 + (self.a1*c2 - self.com2[0]*s2)**2) + self.I2[2, 2]
        M[1, 1] += self.m2 * (self.com2[0]**2) + self.I2[0, 0]
        
        # Add contribution from link 3
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)
        M[0, 0] += self.m3 * ((self.a1*s2 + self.a2*s23 + self.com3[0]*c23)**2 + 
                              (self.a1*c2 + self.a2*c23 - self.com3[0]*s23)**2) + self.I3[2, 2]
        M[1, 1] += self.m3 * ((self.a2 + self.com3[0])**2) + self.I3[0, 0]
        M[2, 2] += self.m3 * (self.com3[0]**2) + self.I3[0, 0]
        
        # Cross terms
        M[0, 1] = self.m2 * (self.com2[0]) * (self.a1*c2) + self.m3 * (self.a2 + self.com3[0]) * (self.a1*c2 + self.a2*c23)
        M[1, 0] = M[0, 1]
        
        M[0, 2] = self.m3 * self.com3[0] * (self.a1*c2 + self.a2*c23)
        M[2, 0] = M[0, 2]
        
        M[1, 2] = self.m3 * self.com3[0] * self.a2
        M[2, 1] = M[1, 2]
        
        return M
    
    def coriolis(self, q, q_dot):
        """Compute the Coriolis and centrifugal forces"""
        q1, q2, q3 = q
        qd1, qd2, qd3 = q_dot
        
        # Simplified Coriolis computation
        C = np.zeros(3)
        
        # Compute trigonometric values
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)
        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)
        
        # First joint
        h = self.m2 * self.a1 * self.com2[0] * s2 + self.m3 * self.a1 * (self.a2 * s2 + self.com3[0] * s23)
        C[0] = -h * (qd2**2 + 2*qd2*qd3 + qd3**2)
        
        # Second joint
        C[1] = self.m2 * self.a1 * self.com2[0] * c2 * qd1**2 + self.m3 * self.a1 * (self.a2 * c2 + self.com3[0] * c23) * qd1**2
        
        # Third joint
        C[2] = self.m3 * self.a1 * self.com3[0] * c23 * qd1**2
        
        return C
    
    def gravity(self, q):
        """Compute the gravity forces"""
        q1, q2, q3 = q
        
        g = np.zeros(3)
        
        # Compute trigonometric values
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)
        
        # Compute gravity components
        g[0] = 0  # No gravity effect on first joint (rotation around vertical axis)
        g[1] = (self.m2 * self.com2[0] + self.m3 * self.a2) * self.gravity[2] * c2 + self.m3 * self.com3[0] * self.gravity[2] * c23
        g[2] = self.m3 * self.com3[0] * self.gravity[2] * c23
        
        return g


class MomentumObserverController:
    """Class to integrate servo control with momentum-based observer"""
    
    def __init__(self, port, baudrate, protocol_version, motor_ids, home_pos):
        """Initialize controller with communication and servo parameters"""
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        self.home_pos = home_pos
        
        # Create robot model
        self.robot = RobotModel(a1=a1, a2=a2, a3=a3)
        
        # PID Controller gains
        self.Kp = np.array([20, 15, 10])  # Proportional gains
        self.Ki = np.array([1, 0.5, 0.2])  # Integral gains
        self.Kd = np.array([5, 2, 1])     # Derivative gains
        
        # Momentum Observer Parameters
        self.K_obs = np.diag([10, 10, 10])  # Observer gain matrix
        
        # Initialize state variables
        self.q_desired = np.zeros(3)  # Desired joint angles (home position)
        self.q = np.zeros(3)          # Current joint angles
        self.q_dot = np.zeros(3)      # Current joint velocities
        self.p_hat = np.zeros(3)      # Initial estimated momentum
        self.tau_ext_hat = np.zeros(3) # Initial estimated external torque
        self.integral_term = np.zeros(3) # Integral term for momentum observer
        self.error_integral = np.zeros(3) # Error integral for PID controller
    
    def connect(self):
        """Connect to the Dynamixel bus using Dynamixel SDK"""
        try:
            # Initialize PortHandler
            self.portHandler = PortHandler(self.port)
            
            # Initialize PacketHandler
            self.packetHandler = PacketHandler(self.protocol_version)
            
            # Open port
            if not self.portHandler.openPort():
                print("Failed to open the port")
                return False
                
            # Set port baudrate
            if not self.portHandler.setBaudRate(self.baudrate):
                print("Failed to change the baudrate")
                return False
                
            # Enable torque for all servos
            for motor_id in self.motor_ids:
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, motor_id, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"Failed to enable torque for Motor ID {motor_id}")
                    print(f"Error: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                    return False
                elif dxl_error != 0:
                    print(f"Error from servo {motor_id}: {self.packetHandler.getRxPacketError(dxl_error)}")
                    return False
            
            print("Successfully connected to Dynamixel bus")
            return True
            
        except Exception as e:
            print(f"Error connecting to Dynamixel bus: {e}")
            return False
    
    def disconnect(self):
        """Disable torque and close port"""
        if hasattr(self, 'portHandler') and self.portHandler.is_open:
            try:
                # Disable torque for all servos
                for motor_id in self.motor_ids:
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, motor_id, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)
                
                # Close port
                self.portHandler.closePort()
                print("Disconnected from Dynamixel bus")
            except Exception as e:
                print(f"Error disconnecting: {e}")
    
    def set_home_position(self):
        """Move all servos to home position"""
        print("Moving all servos to home position...")
        
        for motor_id in self.motor_ids:
            # Write goal position
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, motor_id, ADDR_AX_GOAL_POSITION, self.home_pos)
            
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Failed to write goal position for Motor ID {motor_id}")
                print(f"Error: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Error from servo {motor_id}: {self.packetHandler.getRxPacketError(dxl_error)}")
        
        # Wait for all servos to reach their positions
        time.sleep(2)
        
        # Check if all servos have reached their positions
        all_reached = False
        while not all_reached:
            all_reached = True
            for motor_id in self.motor_ids:
                # Read moving status
                dxl_moving, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                    self.portHandler, motor_id, ADDR_AX_MOVING)
                
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(f"Failed to read moving status from Motor ID {motor_id}")
                    continue
                
                if dxl_moving:
                    all_reached = False
                    break
            
            if not all_reached:
                time.sleep(0.1)
            else:
                print("All servos have reached home position")
                break
    
    def read_servo_position(self, motor_id):
        """Read the current position of a servo"""
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, motor_id, ADDR_AX_PRESENT_POSITION)
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read position from Motor ID {motor_id}")
            print(f"Error: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return self.home_pos, 0.0
        elif dxl_error != 0:
            print(f"Error from servo {motor_id}: {self.packetHandler.getRxPacketError(dxl_error)}")
            return self.home_pos, 0.0
        
        # Convert raw position to angle (in radians)
        # AX-12A: 0-1023 covers -150 ~ 150 degrees, center at 512
        angle = -1 * (dxl_present_position - 512) * (300/1024) * (np.pi/180)
        
        return dxl_present_position, angle
    
    def run_control_loop(self, duration, publish_freq):
        """Run the control loop for momentum-based observer"""
        # Calculate total samples
        total_samples = int(duration * publish_freq)
        period = 1.0 / publish_freq
        
        # Initialize storage arrays for plotting
        self.joint_positions = np.zeros((self.num_motors, total_samples))  # Raw position values (0-1023)
        self.joint_angles = np.zeros((self.num_motors, total_samples))     # Converted angles (radians)
        self.actual_timestamps = np.zeros(total_samples)
        self.target_timestamps = np.zeros(total_samples)
        self.tau_ctrl_history = np.zeros((3, total_samples))
        self.tau_ext_history = np.zeros((3, total_samples))
        self.wrench_history = np.zeros((6, total_samples))  # 6 DOF wrench at end effector
        self.ee_torque_history = np.zeros((3, total_samples)) # End-effector torques (subset of wrench)
        
        # Set up plots for real-time visualization
        self.setup_realtime_plots()
        
        print("Preparing to run the servo control and wrench observer...")
        print(f"Publishing will start in 2 seconds and continue for {duration} seconds.")
        time.sleep(2)
        
        print("Publishing started...")
        
        # Main control loop
        start_time = time.time()
        
        for i in range(total_samples):
            # Calculate target time for this iteration
            next_frame_time = (i * period) + start_time
            self.target_timestamps[i] = i * period
            
            # Read current joint positions from servos
            for m in range(self.num_motors):
                motor_id = self.motor_ids[m]
                
                # Read and process position
                raw_position, angle = self.read_servo_position(motor_id)
                
                # Store raw position and angle
                self.joint_positions[m, i] = raw_position
                self.joint_angles[m, i] = angle
                
                # Update robot model joint position
                self.q[m] = angle
            
            # Estimate joint velocities from position differences
            if i > 0:
                self.q_dot = (self.q - self.joint_angles[:, i-1]) / period
            else:
                self.q_dot = np.zeros(3)
            
            # Get robot dynamics at current state
            M = self.robot.inertia(self.q)
            C = self.robot.coriolis(self.q, self.q_dot)
            G = self.robot.gravity(self.q)
            
            # Calculate control torque (PID)
            error = self.q_desired - self.q
            self.error_integral += error * period
            error_derivative = -self.q_dot
            
            # Apply gains for PID control
            tau_pid = self.Kp * error + self.Ki * self.error_integral + self.Kd * error_derivative
            
            # Complete control torque
            tau_ctrl = tau_pid + G
            
            # Store control torques
            self.tau_ctrl_history[:, i] = tau_ctrl
            
            # Compute current momentum
            p = M @ self.q_dot
            
            # Update the integral term for momentum observer
            integrand = tau_ctrl + C - G + self.tau_ext_hat
            self.integral_term += integrand * period
            
            # Calculate the residual
            r = p - self.integral_term
            
            # Update the estimated external torque
            self.tau_ext_hat = self.K_obs @ r
            
            # Store estimated external torques
            self.tau_ext_history[:, i] = self.tau_ext_hat
            
            # Convert joint space external torque to end-effector wrench
            # Get Jacobian at current configuration
            J = self.robot.jacobian(self.q)
            
            # Compute wrench at end-effector (F = J^-T * tau)
            # Using pseudoinverse for non-square Jacobian
            J_pinv_T = np.transpose(np.linalg.pinv(J))
            wrench = J_pinv_T @ self.tau_ext_hat
            
            # Store wrench data
            self.wrench_history[:, i] = wrench
            
            # Extract and store just the torque components at the end-effector (last 3 components of wrench)
            self.ee_torque_history[:, i] = wrench[3:6]
            
            # Send goal position commands to each servo
            # In this case, we're maintaining the home position
            for m in range(self.num_motors):
                motor_id = self.motor_ids[m]
                
                # Here we're using the home position since we want to maintain it
                self.packetHandler.write2ByteTxRx(
                    self.portHandler, motor_id, ADDR_AX_GOAL_POSITION, self.home_pos)
            
            # Record actual timestamp
            self.actual_timestamps[i] = time.time() - start_time
            
            # Update plots periodically
            if i % 5 == 0 or i == 0 or i == total_samples - 1:
                self.update_plots(i)
            
            # Wait until we reach the next target time
            current_time = time.time()
            if current_time < next_frame_time:
                time.sleep(next_frame_time - current_time)
        
        # Return all servos to center position at the end
        self.set_home_position()
        
        # Plot final results
        self.plot_final_results()
        
        # Save data
        np.savez('momentum_observer_data.npz', 
                 target_timestamps=self.target_timestamps,
                 joint_positions=self.joint_positions,
                 joint_angles=self.joint_angles,
                 tau_ctrl_history=self.tau_ctrl_history,
                 tau_ext_history=self.tau_ext_history,
                 wrench_history=self.wrench_history,
                 ee_torque_history=self.ee_torque_history,
                 motor_ids=self.motor_ids)
        
        print("Data saved to momentum_observer_data.npz")
    
    def setup_realtime_plots(self):
        """Set up figures for real-time plotting"""
        # Plot 1: Joint Positions
        self.fig_joint, self.ax_joint = plt.subplots(figsize=(10, 6))
        self.h_joint = []
        color_codes = ['r', 'g', 'b']
        
        for i in range(self.num_motors):
            line, = self.ax_joint.plot([], [], color_codes[i] + '-', linewidth=1.5)
            self.h_joint.append(line)
        
        self.ax_joint.set_xlabel('Time (seconds)')
        self.ax_joint.set_ylabel('Joint Position (rad)')
        self.ax_joint.set_title('Joint Positions')
        self.ax_joint.grid(True)
        self.ax_joint.set_ylim(-1, 1)
        self.ax_joint.legend([f'Joint {i+1}' for i in range(self.num_motors)])
        
        # Plot 2: End-Effector External Torques
        self.fig_ee_torque, self.ax_ee_torque = plt.subplots(figsize=(10, 6))
        self.h_ee_torque = []
        torque_colors = ['r', 'g', 'b']
        
        for i in range(3):
            line, = self.ax_ee_torque.plot([], [], torque_colors[i] + '-', linewidth=1.5)
            self.h_ee_torque.append(line)
        
        self.ax_ee_torque.set_xlabel('Time (seconds)')
        self.ax_ee_torque.set_ylabel('Torque (Nm)')
        self.ax_ee_torque.set_title('Estimated External Torques at End-Effector')
        self.ax_ee_torque.grid(True)
        self.ax_ee_torque.legend(['T_x', 'T_y', 'T_z'])
        
        # Plot 3: Control Torques
        self.fig_tau, self.ax_tau = plt.subplots(figsize=(10, 6))
        self.h_tau = []
        
        for i in range(self.num_motors):
            line, = self.ax_tau.plot([], [], color_codes[i] + '-', linewidth=1.5)
            self.h_tau.append(line)
        
        self.ax_tau.set_xlabel('Time (seconds)')
        self.ax_tau.set_ylabel('Torque (Nm)')
        self.ax_tau.set_title('PID Controller Torques')
        self.ax_tau.grid(True)
        self.ax_tau.legend([f'Joint {i+1}' for i in range(self.num_motors)])
        
        # Initial display
        plt.ion()  # Turn on interactive mode
        plt.pause(0.1)
    
    def update_plots(self, index):
        """Update real-time plots with new data"""
        # Update joint positions plot
        for j in range(self.num_motors):
            self.h_joint[j].set_xdata(self.target_timestamps[:index+1])
            self.h_joint[j].set_ydata(self.joint_angles[j, :index+1])
        
        self.ax_joint.relim()
        self.ax_joint.autoscale_view()
        
        # Update end-effector torque plot
        for j in range(3):
            self.h_ee_torque[j].set_xdata(self.target_timestamps[:index+1])
            self.h_ee_torque[j].set_ydata(self.ee_torque_history[j, :index+1])
        
        self.ax_ee_torque.relim()
        self.ax_ee_torque.autoscale_view()
        
        # Update control torques plot
        for j in range(self.num_motors):
            self.h_tau[j].set_xdata(self.target_timestamps[:index+1])
            self.h_tau[j].set_ydata(self.tau_ctrl_history[j, :index+1])
        
        self.ax_tau.relim()
        self.ax_tau.autoscale_view()
        
        # Redraw
        self.fig_joint.canvas.draw_idle()
        self.fig_ee_torque.canvas.draw_idle()
        self.fig_tau.canvas.draw_idle()
        plt.pause(0.01)
    
    def plot_final_results(self):
        """Create final plots with comprehensive visualizations"""
        plt.ioff()  # Turn off interactive mode
        
        # Plot 1: Joint Positions
        fig, axes = plt.subplots(self.num_motors, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Joint Positions (Final)')
        
        for j in range(self.num_motors):
            axes[j].plot(self.target_timestamps, self.joint_angles[j, :], 'b-', linewidth=1.5)
            axes[j].axhline(y=0, color='r', linestyle='--', label='Home Position')
            axes[j].set_ylabel(f'Joint {j+1} (rad)')
            axes[j].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        # Plot 2: Estimated External Torques at End-Effector
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('End-Effector External Torques (Final)')
        
        for j in range(3):
            axes[j].plot(self.target_timestamps, self.ee_torque_history[j, :], 'b-', linewidth=1.5)
            axes[j].axhline(y=0, color='k', linestyle='--')
            axes[j].set_ylabel(f'T_{"xyz"[j]} (Nm)')
            axes[j].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        # Plot 3: Complete Wrench at End-Effector
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('End-Effector Wrench (Final)')
        
        # Forces
        axes[0].plot(self.target_timestamps, self.wrench_history[0, :], 'r-', linewidth=1.5, label='F_x')
        axes[0].plot(self.target_timestamps, self.wrench_history[1, :], 'g-', linewidth=1.5, label='F_y')
        axes[0].plot(self.target_timestamps, self.wrench_history[2, :], 'b-', linewidth=1.5, label='F_z')
        axes[0].set_title('Forces at End-Effector')
        axes[0].set_ylabel('Force (N)')
        axes[0].grid(True)
        axes[0].legend()
        
        # Torques
        axes[1].plot(self.target_timestamps, self.wrench_history[3, :], 'r-', linewidth=1.5, label='T_x')
        axes[1].plot(self.target_timestamps, self.wrench_history[4, :], 'g-', linewidth=1.5, label='T_y')
        axes[1].plot(self.target_timestamps, self.wrench_history[5, :], 'b-', linewidth=1.5, label='T_z')
        axes[1].set_title('Torques at End-Effector')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Torque (Nm)')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot 4: Control Torques (PID)
        fig, axes = plt.subplots(self.num_motors, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Control Torques (Final)')
        
        for j in range(self.num_motors):
            axes[j].plot(self.target_timestamps, self.tau_ctrl_history[j, :], 'b-', linewidth=1.5)
            axes[j].axhline(y=0, color='k', linestyle='--')
            axes[j].set_ylabel(f'Joint {j+1} (Nm)')
            axes[j].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        plt.show()


# Main execution code
if __name__ == "__main__":
    try:
        # Create servo controller with momentum observer
        controller = MomentumObserverController(
            port=PORT,
            baudrate=BAUDRATE,
            protocol_version=PROTOCOL_VERSION,
            motor_ids=MOTOR_IDS,
            home_pos=HOME_POS
        )
        
        # Connect to servos
        if controller.connect():
            # Set initial home position
            controller.set_home_position()
            
            # Run control loop
            controller.run_control_loop(duration=DURATION, publish_freq=PUBLISH_FREQ)
            
            # Disconnect when done
            controller.disconnect()
        else:
            print("Failed to connect to servos. Check port and connections.")
    
    except Exception as e:
        print(f"Error: {e}")
        
        # Clean up connection if error occurs
        if 'controller' in locals() and hasattr(controller, 'portHandler'):
            controller.disconnect()

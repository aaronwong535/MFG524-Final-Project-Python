% AX-12A Servo Control with Momentum-Based Wrench Observer
% Integrating servo control with a momentum-based observer for detecting external wrenches
% This script maintains servos at the home position while monitoring for external forces

close all
clear all
clc

% Configuration parameters
PORT = 'COM5';      % Serial port - update as needed
MOTOR_IDS = [1, 2, 3];  % Three motor IDs
NUM_MOTORS = length(MOTOR_IDS);
PUBLISH_FREQ = 15;  % Hz
DURATION = 10;      % Seconds
TOTAL_SAMPLES = PUBLISH_FREQ * DURATION;

% Position parameters - keeping robot at home position
HOME_POS = 512;     % Center position (0-1023) for all servos

%% Robot Model Setup (from Momentum_based_Observer.m)
% Link lengths
a1 = 0.055;
a2 = 0.0675;
a3 = 0.04;

% Create a robot model using Robotics System Toolbox
robot = rigidBodyTree('DataFormat', 'column');

% Define robot base
base = robot.Base;

% Create link 1
body1 = rigidBody('link1');
joint1 = rigidBodyJoint('joint1', 'revolute');
setFixedTransform(joint1, trvec2tform([0 0 0]));
joint1.JointAxis = [0 0 1]; % Rotation around z-axis
body1.Joint = joint1;
body1.Mass = 1;
body1.CenterOfMass = [0 0 a1/2];
body1.Inertia = [0.01 0.01 0.01 0 0 0]; % Simplified inertia

% Add first link to robot
addBody(robot, body1, 'base');

% Create link 2
body2 = rigidBody('link2');
joint2 = rigidBodyJoint('joint2', 'revolute');
setFixedTransform(joint2, trvec2tform([0 0 a1]) * rotm2tform([1 0 0; 0 0 -1; 0 1 0]));
joint2.JointAxis = [0 0 1]; % Rotation around z-axis
body2.Joint = joint2;
body2.Mass = 1;
body2.CenterOfMass = [a2/2 0 0];
body2.Inertia = [0.01 0.01 0.01 0 0 0]; % Simplified inertia

% Add second link to robot
addBody(robot, body2, 'link1');

% Create link 3
body3 = rigidBody('link3');
joint3 = rigidBodyJoint('joint3', 'revolute');
setFixedTransform(joint3, trvec2tform([a2 0 0]));
joint3.JointAxis = [0 0 1]; % Rotation around z-axis
body3.Joint = joint3;
body3.Mass = 1;
body3.CenterOfMass = [a3/2 0 0];
body3.Inertia = [0.01 0.01 0.01 0 0 0]; % Simplified inertia

% Add third link to robot
addBody(robot, body3, 'link2');

% Create end-effector frame
endEffector = rigidBody('endEffector');
endEffectorJoint = rigidBodyJoint('endEffectorJoint', 'fixed');
setFixedTransform(endEffectorJoint, trvec2tform([a3 0 0]) * rotm2tform([0 0 1; 1 0 0; 0 1 0]));
endEffector.Joint = endEffectorJoint;

% Add end-effector to robot
addBody(robot, endEffector, 'link3');

robot.Gravity = [0 0 -9.81]';

%% Controller Parameters
% PID Controller gains - convert to column vectors
Kp = [20; 15; 10]; % Proportional gains
Ki = [1; 0.5; 0.2]; % Integral gains
Kd = [5; 2; 1];    % Derivative gains

% Momentum Observer Parameters
K_obs = diag([10, 10, 10]); % Observer gain matrix

% Initial conditions
q_desired = [0; 0; 0];   % Desired joint angles (home position)
q = zeros(3, 1);         % Current joint angles
q_dot = zeros(3, 1);     % Current joint velocities
p_hat = zeros(3, 1);     % Initial estimated momentum
tau_ext_hat = zeros(3, 1); % Initial estimated external torque
integral_term = zeros(3, 1); % Integral term for momentum observer
error_integral = zeros(3, 1); % Error integral for PID controller

% Initialize storage arrays for plotting
joint_positions = zeros(NUM_MOTORS, TOTAL_SAMPLES);  % Raw position values (0-1023)
joint_angles = zeros(NUM_MOTORS, TOTAL_SAMPLES);     % Converted angles (radians)
actual_timestamps = zeros(1, TOTAL_SAMPLES);
target_timestamps = zeros(1, TOTAL_SAMPLES);
tau_ctrl_history = zeros(3, TOTAL_SAMPLES);
tau_ext_history = zeros(3, TOTAL_SAMPLES);
wrench_history = zeros(6, TOTAL_SAMPLES);  % 6 DOF wrench at end effector
ee_torque_history = zeros(3, TOTAL_SAMPLES); % End-effector torques (subset of wrench)

% Setup figures for real-time plotting
figure('Name', 'Joint Positions');
h_joint = zeros(NUM_MOTORS, 1);
colorCodes = {'r', 'g', 'b'};
for i = 1:NUM_MOTORS
    h_joint(i) = plot(0, 0, [colorCodes{i}, '-'], 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (seconds)');
ylabel('Joint Position (rad)');
title('Joint Positions');
grid on;
ylim([-1, 1]);  % Assuming small movements around home position
legend('Joint 1', 'Joint 2', 'Joint 3');

figure('Name', 'End-Effector External Torques');
h_ee_torque = zeros(3, 1);
torqueColors = {'r', 'g', 'b'};
for i = 1:3
    h_ee_torque(i) = plot(0, 0, [torqueColors{i}, '-'], 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (seconds)');
ylabel('Torque (Nm)');
title('Estimated External Torques at End-Effector');
grid on;
legend('T_x', 'T_y', 'T_z');

figure('Name', 'Control Torques');
h_tau = zeros(NUM_MOTORS, 1);
for i = 1:NUM_MOTORS
    h_tau(i) = plot(0, 0, [colorCodes{i}, '-'], 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (seconds)');
ylabel('Torque (Nm)');
title('PID Controller Torques');
grid on;
legend('Joint 1', 'Joint 2', 'Joint 3');

% Initialize serial port
try
    disp('Initializing serial port...');
    serialObj = serialport(PORT, 1000000, 'DataBits', 8, 'Parity', 'none', 'StopBits', 1);
    serialObj.Timeout = 0.05;  % Short timeout to avoid blocking
    
    % Define constants for AX-12A communication
    INST_WRITE = 3;        % Write instruction
    INST_READ = 2;         % Read instruction
    GOAL_POS_ADDR = 30;    % Address for Goal Position
    PRESENT_POS_ADDR = 36; % Address for Present Position
    PRESENT_SPEED_ADDR = 38; % Address for Present Speed
    HEADER = [255, 255];   % Header bytes
    
    % Move all servos to home position before starting
    disp('Moving all servos to home position...');
    for motor = 1:NUM_MOTORS
        motorID = MOTOR_IDS(motor);
        posL = bitand(HOME_POS, 255);  % Lower byte
        posH = bitshift(HOME_POS, -8); % Upper byte
        packetLength = 5;  % ID + Instruction + Parameter1 + Parameter2 + Parameter3
        checksum = mod(bitcmp(uint8(motorID + packetLength + INST_WRITE + GOAL_POS_ADDR + posL + posH)), 256);
        packet = [HEADER, motorID, packetLength, INST_WRITE, GOAL_POS_ADDR, posL, posH, checksum];
        
        % Send command
        write(serialObj, packet, "uint8");
        
        % Wait for servo to respond
        try
            response = read(serialObj, 6, "uint8");
        catch
            % Continue if no response in time
        end
    end
    
    % Wait for servos to reach home position
    disp('Waiting for servos to reach home position...');
    pause(2);
    
    % Create read instruction packets for each motor (like in AX12AReader_3.m)
    read_packets = cell(NUM_MOTORS, 1);
    for m = 1:NUM_MOTORS
        motorID = MOTOR_IDS(m);
        packetLength = 4;   % ID + Instruction + Parameter1 + Parameter2
        checksum = mod(bitcmp(uint8(motorID + packetLength + INST_READ + PRESENT_POS_ADDR + 2)), 256);
        read_packets{m} = [HEADER, motorID, packetLength, INST_READ, PRESENT_POS_ADDR, 2, checksum];
    end
    
    disp('Preparing to run the servo control and wrench observer...');
    disp(['Publishing will start in 2 seconds and continue for ', num2str(DURATION), ' seconds.']);
    pause(2);
    
    disp('Publishing started...');
    
    % Flush input buffer
    flush(serialObj);
    
    % Set up timing using absolute time approach
    startTime = tic;
    nextFrameTime = 0;
    period = 1/PUBLISH_FREQ;
    
    % Main control loop
    for i = 1:TOTAL_SAMPLES
        % Calculate absolute target time for this iteration
        nextFrameTime = (i-1) * period;
        target_timestamps(i) = nextFrameTime;
        
        % Read current joint positions from servos (using code from AX12AReader_3.m)
        for m = 1:NUM_MOTORS
            % Send read position packet
            write(serialObj, read_packets{m}, "uint8");
            
            % Read response
            try
                response = read(serialObj, 8, "uint8");
                
                % Process response
                if length(response) >= 8
                    error_byte = response(5);
                    if error_byte == 0
                        % Extract position data (2 bytes)
                        positionL = response(6);    % Lower byte
                        positionH = response(7);    % Upper byte
                        
                        % Combine bytes (L + H*256)
                        rawPosition = positionL + positionH * 256;
                        
                        % Store raw position
                        joint_positions(m, i) = rawPosition;
                        
                        % Convert position to angle (in radians, not degrees)
                        % Use negative to correct the reversed joint position
                        angle = -1 * (rawPosition - 512) * (pi/512);
                        joint_angles(m, i) = angle;
                        
                        % Update robot model joint position
                        q(m) = angle;
                    else
                        % Use previous value if there's an error
                        if i > 1
                            joint_positions(m, i) = joint_positions(m, i-1);
                            joint_angles(m, i) = joint_angles(m, i-1);
                        else
                            joint_positions(m, i) = HOME_POS;
                            joint_angles(m, i) = 0;
                        end
                        disp(['Warning: Error code from Servo ', num2str(MOTOR_IDS(m)), ': ', num2str(error_byte)]);
                    end
                else
                    % Use previous value if incomplete response
                    if i > 1
                        joint_positions(m, i) = joint_positions(m, i-1);
                        joint_angles(m, i) = joint_angles(m, i-1);
                    else
                        joint_positions(m, i) = HOME_POS;
                        joint_angles(m, i) = 0;
                    end
                    disp(['Warning: Incomplete response from Servo ', num2str(MOTOR_IDS(m))]);
                end
            catch
                % Use previous value if read fails
                if i > 1
                    joint_positions(m, i) = joint_positions(m, i-1);
                    joint_angles(m, i) = joint_angles(m, i-1);
                else
                    joint_positions(m, i) = HOME_POS;
                    joint_angles(m, i) = 0;
                end
                disp(['Warning: Failed to read from Servo ', num2str(MOTOR_IDS(m))]);
            end
            
            % Small delay between motor readings
            pause(0.01);
        end
        
        % Estimate joint velocities from position differences
        if i > 1
            q_dot = (q - [joint_angles(1, i-1); joint_angles(2, i-1); joint_angles(3, i-1)]) / period;
        else
            q_dot = zeros(3, 1);
        end
        
        % Get robot dynamics at current state
        M = massMatrix(robot, q);
        C = velocityProduct(robot, q, q_dot);
        G = gravityTorque(robot, q);
        
        % Calculate control torque (PID)
        error = q_desired - q;
        error_integral = error_integral + error * period;
        error_derivative = -q_dot;
        
        % Apply element-wise multiplication
        tau_pid = zeros(3, 1);
        for j = 1:3
            tau_pid(j) = Kp(j) * error(j) + Ki(j) * error_integral(j) + Kd(j) * error_derivative(j);
        end
        
        tau_ctrl = tau_pid + G;
        
        % Store control torques
        tau_ctrl_history(:, i) = tau_ctrl;
        
        % Compute current momentum
        p = M * q_dot;
        
        % Update the integral term for momentum observer
        integrand = tau_ctrl + C - G + tau_ext_hat;
        integral_term = integral_term + integrand * period;
        
        % Calculate the residual
        r = p - integral_term;
        
        % Update the estimated external torque
        tau_ext_hat = K_obs * r;
        
        % Store estimated external torques
        tau_ext_history(:, i) = tau_ext_hat;
        
        % Convert joint space external torque to end-effector wrench
        % Get Jacobian at current configuration
        J = geometricJacobian(robot, q, 'endEffector');
        
        % Compute wrench at end-effector (F = J^-T * tau)
        % Using pseudoinverse for non-square Jacobian
        wrench = pinv(J)' * tau_ext_hat;
        
        % Store wrench data
        wrench_history(:, i) = wrench;
        
        % Extract and store just the torque components at the end-effector (last 3 components of wrench)
        ee_torque_history(:, i) = wrench(4:6);
        
        % Send goal position commands to each servo
        % In this case, we're maintaining the home position
        for motor = 1:NUM_MOTORS
            motorID = MOTOR_IDS(motor);
            
            % Compute position command from PID controller
            % Here we're using the home position since we want to maintain it
            posCommand = HOME_POS;
            
            % Create write position packet
            posL = bitand(posCommand, 255);  % Lower byte
            posH = bitshift(posCommand, -8); % Upper byte
            packetLength = 5;  % ID + Instruction + Parameter1 + Parameter2 + Parameter3
            checksum = mod(bitcmp(uint8(motorID + packetLength + INST_WRITE + GOAL_POS_ADDR + posL + posH)), 256);
            packet = [HEADER, motorID, packetLength, INST_WRITE, GOAL_POS_ADDR, posL, posH, checksum];
            
            % Send command
            write(serialObj, packet, "uint8");
            
            % Try to read status packet quickly without blocking too long
            try
                response = read(serialObj, 6, "uint8");
            catch
                % Just continue if no response in time
            end
        end
        
        % Record actual timestamp
        actual_timestamps(i) = toc(startTime);
        
        % Update plots periodically
        if mod(i, 5) == 0 || i == 1 || i == TOTAL_SAMPLES
            % Update joint positions plot
            for j = 1:NUM_MOTORS
                set(h_joint(j), 'XData', target_timestamps(1:i), 'YData', joint_angles(j, 1:i));
            end
            
            % Update end-effector torque plot
            for j = 1:3
                set(h_ee_torque(j), 'XData', target_timestamps(1:i), 'YData', ee_torque_history(j, 1:i));
            end
            
            % Update control torques plot
            for j = 1:NUM_MOTORS
                set(h_tau(j), 'XData', target_timestamps(1:i), 'YData', tau_ctrl_history(j, 1:i));
            end
            
            drawnow limitrate;
        end
        
        % Wait until we reach the next target time
        while toc(startTime) < nextFrameTime + period
            % Use a shorter pause for more precise timing
            pause(0.001);
        end
    end
    
    % Return all servos to center position at the end
    disp('Returning all servos to center position...');
    
    for motor = 1:NUM_MOTORS
        motorID = MOTOR_IDS(motor);
        posL = bitand(HOME_POS, 255);
        posH = bitshift(HOME_POS, -8);
        packetLength = 5;
        checksum = mod(bitcmp(uint8(motorID + packetLength + INST_WRITE + GOAL_POS_ADDR + posL + posH)), 256);
        packet = [HEADER, motorID, packetLength, INST_WRITE, GOAL_POS_ADDR, posL, posH, checksum];
        write(serialObj, packet, "uint8");
        pause(0.1); % Small delay between commands
    end
    
    disp('All servos returned to center position.');
    
    % Close serial connection
    clear serialObj;
    
    % Create final plots with better visualizations
    % Plot 1: Joint Positions
    figure('Name', 'Joint Positions (Final)');
    for j = 1:NUM_MOTORS
        subplot(NUM_MOTORS, 1, j);
        plot(target_timestamps, joint_angles(j, :), 'b-', 'LineWidth', 1.5);
        hold on;
        yline(0, 'r--', 'Home Position');
        ylabel(['Joint ' num2str(j) ' (rad)']);
        if j == 1
            title('Joint Positions');
        end
        if j == NUM_MOTORS
            xlabel('Time (s)');
        end
        grid on;
    end
    
    % Plot 2: Estimated External Torques at End-Effector
    figure('Name', 'End-Effector External Torques (Final)');
    for j = 1:3
        subplot(3, 1, j);
        plot(target_timestamps, ee_torque_history(j, :), 'b-', 'LineWidth', 1.5);
        hold on;
        yline(0, 'k--');
        ylabel(['T_' 'xyz(j)' ' (Nm)']);
        if j == 1
            title('Estimated External Torques at End-Effector');
        end
        if j == 3
            xlabel('Time (s)');
        end
        grid on;
    end
    
    % Plot 3: Complete Wrench at End-Effector
    figure('Name', 'End-Effector Wrench (Final)');
    % Forces
    subplot(2, 1, 1);
    plot(target_timestamps, wrench_history(1:3, :)', 'LineWidth', 1.5);
    title('Forces at End-Effector');
    ylabel('Force (N)');
    grid on;
    legend('F_x', 'F_y', 'F_z');
    
    % Torques
    subplot(2, 1, 2);
    plot(target_timestamps, wrench_history(4:6, :)', 'LineWidth', 1.5);
    title('Torques at End-Effector');
    xlabel('Time (s)');
    ylabel('Torque (Nm)');
    grid on;
    legend('T_x', 'T_y', 'T_z');
    
    % Plot 4: Control Torques (PID)
    figure('Name', 'Control Torques (Final)');
    for j = 1:NUM_MOTORS
        subplot(NUM_MOTORS, 1, j);
        plot(target_timestamps, tau_ctrl_history(j, :), 'b-', 'LineWidth', 1.5);
        hold on;
        yline(0, 'k--');
        ylabel(['Joint ' num2str(j) ' (Nm)']);
        if j == 1
            title('Control Torques (PID Controller)');
        end
        if j == NUM_MOTORS
            xlabel('Time (s)');
        end
        grid on;
    end
    
    % Save data to file
    save('momentum_observer_data.mat', 'target_timestamps', 'joint_positions', 'joint_angles', ...
        'tau_ctrl_history', 'tau_ext_history', 'wrench_history', 'ee_torque_history', 'MOTOR_IDS');
    disp('Data saved to momentum_observer_data.mat');
    
catch exception
    disp(['Error: ', exception.message]);
    
    % Clean up serial connection if error occurs
    if exist('serialObj', 'var')
        clear serialObj;
    end
end
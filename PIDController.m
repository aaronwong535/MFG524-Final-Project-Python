close all
clear all
clc

% 1. Define Robot Model (using Robotics System Toolbox)
robot = rigidBodyTree('DataFormat', 'column');

% Link lengths
a1 = 0.055;
a2 = 0.0675;
a3 = 0.04;

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
% robot.Gravity = [0 0 0]';

%% 2. Define PID Controller Parameters
% PID gains for each joint
Kp = [40, 40, 20]; % Proportional gains
Ki = [1, 0.5, 0.2]; % Integral gains
Kd = [10, 2, 2];    % Derivative gains

%% 3. Define Desired Pose and Initial Conditions
% Initial joint angles and velocities
q0 = [0; 0; 0];
q_dot0 = [0; 0; 0];

% Desired joint angles (target pose)
q_desired = [0; 0; 0];

%% 4. Set Simulation Parameters
t_start = 0;
t_end = 3;
dt = 0.01;
tspan = t_start:dt:t_end;
num_steps = length(tspan);
log_dt = 0.1; % 10 Hz = 0.1 seconds
log_steps = floor(t_end / log_dt) + 1;
t_log = t_start:log_dt:t_end;

%% 5. Initialize Variables for Simulation
q_history = zeros(num_steps, 3);
q_dot_history = zeros(num_steps, 3);
tau_history = zeros(num_steps, 3);
error_history = zeros(num_steps, 3);
error_integral = zeros(3, 1);
q_log_history = zeros(log_steps, 3); % For logging at 10 Hz

q = q0;
q_dot = q_dot0;
q_history(1, :) = q';
q_dot_history(1, :) = q_dot';
q_log_history(1, :) = q';

%% 6. Main Simulation Loop with PID Controller
log_counter = 1;
next_log_time = t_start + log_dt;

for i = 2:num_steps
    current_time = tspan(i);
    dt = tspan(i) - tspan(i-1);
    
    % Calculate error terms
    error = q_desired - q;
    error_integral = error_integral + error * dt;
    error_derivative = -q_dot;
    
    % PID Control Law
    tau = Kp' .* error + Ki' .* error_integral + Kd' .* error_derivative;
    tau_max = 20;
    tau = min(max(tau, -tau_max), tau_max);
    
    error_history(i, :) = error';
    
    % Simulate dynamics
    [t_temp, state_temp] = ode113(@(t, state) manipulator_dynamics(t, state, robot, tau), [tspan(i-1), tspan(i)], [q; q_dot]);
    
    q = state_temp(end, 1:3)';
    q_dot = state_temp(end, 4:6)';
    
    q_history(i, :) = q';
    q_dot_history(i, :) = q_dot';
    tau_history(i, :) = tau';

    % Log at 10 Hz
    if current_time >= next_log_time
        log_counter = log_counter + 1;
        q_log_history(log_counter, :) = q';
        next_log_time = next_log_time + log_dt;
    end
end

% 7. Calculate End-Effector Position Over Time
ee_positions = zeros(num_steps, 3);
ee_positions_log = zeros(log_steps, 3); % For 10 Hz logged positions
for i = 1:num_steps
    T = getTransform(robot, q_history(i,:)', 'endEffector', 'base');
    ee_positions(i,:) = T(1:3, 4)';
end
for i = 1:log_steps
    T = getTransform(robot, q_log_history(i,:)', 'endEffector', 'base');
    ee_positions_log(i,:) = T(1:3, 4)';
end

% 8. Plot Results
% Joint Positions Plot
figure;
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, q_history(:, j), 'b-', 'DisplayName', 'Simulated');
    hold on;
    plot(t_log, q_log_history(:, j), 'g-o', 'DisplayName', 'Logged (10 Hz)');
    plot(tspan, q_desired(j) * ones(size(tspan)), 'r--', 'DisplayName', 'Desired');
    ylabel(['Joint ' num2str(j) ' (rad)']);
    legend;
    if j == 1
        title('Joint Positions');
    end
    if j == 3
        xlabel('Time (s)');
    end
end

% Enhanced End-Effector Trajectory Plot
figure;
% Plot full trajectory
plot3(ee_positions(:,1), ee_positions(:,2), ee_positions(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Full Trajectory');
hold on;
% Plot 10 Hz logged points
plot3(ee_positions_log(:,1), ee_positions_log(:,2), ee_positions_log(:,3), 'k*', 'MarkerSize', 6, 'DisplayName', '10 Hz Points');
% Plot start and end points
plot3(ee_positions(1,1), ee_positions(1,2), ee_positions(1,3), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start');
plot3(ee_positions(end,1), ee_positions(end,2), ee_positions(end,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End');
grid on;
xlabel('X (m)', 'FontSize', 12);
ylabel('Y (m)', 'FontSize', 12);
zlabel('Z (m)', 'FontSize', 12);
title('End-Effector Trajectory', 'FontSize', 14);
legend('Location', 'best');
view(-37.5, 30); % Adjust view angle for better visualization
axis equal; % Equal scaling on all axes

%% Define the system's equations of motion as a function
function dqdt = manipulator_dynamics(t, state, robot, tau)
    q = state(1:3);
    q_dot = state(4:6);

    M = massMatrix(robot, q);
    C = velocityProduct(robot, q, q_dot);
    G = gravityTorque(robot, q);

    q_ddot = M\(tau - C - G);

    dqdt = [q_dot; q_ddot];
end
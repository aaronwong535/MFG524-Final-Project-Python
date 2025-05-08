close all
clear all
clc

%% MOMENTUM-BASED WRENCH OBSERVER IMPLEMENTATION
% Based on the slides and reference files provided

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
body1.Mass = 0.2;
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
body2.Mass = 0.2;
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
body3.Mass = 0.2;
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

%% 2. Define Controller Parameters
Kp = [20, 15, 10]; % Proportional gains
Ki = [1, 0.5, 0.2]; % Integral gains
Kd = [5, 2, 1];    % Derivative gains

%% 3. Momentum Observer Parameters
% Observer gain matrix (diagonal matrix for simplicity)
K_obs = diag([2, 2, 2]); % Lower gains to reduce oscillations

% Initial conditions
q0 = [0; 0; 0];      % Initial joint angles
q_dot0 = [0; 0; 0];  % Initial joint velocities
p_hat = zeros(3, 1); % Initial estimated momentum
tau_ext_hat = zeros(3, 1); % Initial estimated external torque
integral_term = zeros(3, 1); % NEW: Add integral term for the formula

%% 4. Simulation Parameters
t_start = 0;
t_end = 5;
dt = 0.01;
tspan = t_start:dt:t_end;
num_steps = length(tspan);

% Target position (holding position)
q_desired = [0; 0; 0];

%% 5. Define External Force Profile (time-varying force applied to end effector)
% We'll apply a force in the x-direction starting at t=2s and ending at t=4s
mass = 0.1;
g = 9.81;
F_ext_magnitude = mass * g; % N
F_ext_direction = [0; 0; -1]; % Unit vector in x-direction
F_ext_start = 1.5; % start time (seconds)
F_ext_end = 3.5;   % end time (seconds)

%% 6. Initialize Storage Variables
q_history = zeros(num_steps, 3);
q_dot_history = zeros(num_steps, 3);
tau_history = zeros(num_steps, 3);
tau_ext_history = zeros(num_steps, 3);
p_hat_history = zeros(num_steps, 3);
r_history = zeros(num_steps, 3);  % Residual
error_history = zeros(num_steps, 3);
error_integral = zeros(3, 1);
F_ext_applied_history = zeros(num_steps, 3);

q = q0;
q_dot = q_dot0;
q_history(1, :) = q';
q_dot_history(1, :) = q_dot';

%% 7. Main Simulation Loop with Momentum Observer
% Calculate initial mass matrix and momentum
M_initial = massMatrix(robot, q0);
p = M_initial * q_dot0;  % Initialize momentum
p_hat_history(1, :) = p';

% Convert gain vectors to column vectors
Kp = Kp(:);  % Convert to column vector
Ki = Ki(:);  % Convert to column vector
Kd = Kd(:);  % Convert to column vector

for i = 2:num_steps
    current_time = tspan(i);
    dt = tspan(i) - tspan(i-1);
    
    % Get robot dynamics at current state
    M = massMatrix(robot, q);
    C = velocityProduct(robot, q, q_dot);
    G = gravityTorque(robot, q);
    
    % Calculate control torque (PID)
    error = q_desired - q;
    error_integral = error_integral + error * dt;
    error_derivative = -q_dot;
    
    tau_pid = Kp .* error + Ki .* error_integral + Kd .* error_derivative;
    tau_ctrl = tau_pid + G;
    error_history(i, :) = error';
    
    % Compute current momentum
    p = M * q_dot;
    
    % MODIFIED: Update the integral term according to the formula
    % integrand = u + C - G + ŵ_e
    integrand = tau_ctrl + C - G + tau_ext_hat;
    integral_term = integral_term + integrand * dt;
    
    % MODIFIED: Calculate the residual according to the formula
    % r = p - ∫(...)dt
    r = p - integral_term;
    
    % MODIFIED: Update the estimated external torque directly using the formula
    % ŵ_e = K_M * r
    tau_ext_hat = K_obs * r;
    
    % Store residual
    r_history(i, :) = r';  % Transpose to match history array dimensions
    
    % Apply external force (if within the time window)
    F_ext = zeros(3, 1);
    if current_time >= F_ext_start && current_time <= F_ext_end
        F_ext = F_ext_magnitude * F_ext_direction;
    end
    
    % Convert external force to joint torques using the Jacobian
    J = geometricJacobian(robot, q, 'endEffector');
    J_linear = J(4:6,:); % Extract the linear part (for force)
    tau_ext = J_linear' * F_ext;
    
    F_ext_applied_history(i, :) = F_ext';
    
    % Combine control and external torques
    tau_total = tau_ctrl + tau_ext;
    
    % Simulate dynamics
    [~, state_temp] = ode113(@(t, state) manipulator_dynamics(t, state, robot, tau_total), [tspan(i-1), tspan(i)], [q; q_dot]);
    
    q = state_temp(end, 1:3)';
    q_dot = state_temp(end, 4:6)';
    
    % Store history
    q_history(i, :) = q';
    q_dot_history(i, :) = q_dot';
    tau_history(i, :) = tau_ctrl';
    tau_ext_history(i, :) = tau_ext_hat';
    p_hat_history(i, :) = p';
end
%% 8. Calculate End-Effector Positions Over Time
ee_positions = zeros(num_steps, 3);
for i = 1:num_steps
    T = getTransform(robot, q_history(i,:)', 'endEffector', 'base');
    ee_positions(i,:) = T(1:3, 4)';
end

%% 9. Plot Results
% Plot 1: Joint Positions
figure('Name', 'Joint Positions');
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, q_history(:, j), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(tspan, q_desired(j) * ones(size(tspan)), 'r--', 'LineWidth', 1);
    ylabel(['Joint ' num2str(j) ' (rad)']);
    if j == 1
        title('Joint Positions');
    end
    if j == 3
        xlabel('Time (s)');
    end
    grid on;
end

% Plot 2: Estimated External Torques (main result)
figure('Name', 'Estimated External Torques (Joint Space)');
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, tau_ext_history(:, j), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Plot the time window when external force is applied
    yline(0, 'k--');
    xline(F_ext_start, 'g--', 'Force Applied');
    xline(F_ext_end, 'r--', 'Force Removed');
    
    ylabel(['Joint ' num2str(j) ' (Nm)']);
    if j == 1
        title('Estimated External Torques (Wrench Observer)');
    end
    if j == 3
        xlabel('Time (s)');
    end
    grid on;
    legend('Estimated Torque');
end

% Plot 3: External Force Applied
figure('Name', 'Applied External Force');
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, F_ext_applied_history(:, j), 'r-', 'LineWidth', 1.5);
    
    ylabel(['Force ' 'xyz(j)' ' (N)']);
    if j == 1
        title('External Force Applied to End Effector');
    end
    if j == 3
        xlabel('Time (s)');
    end
    grid on;
end

% Plot 4: End-Effector Trajectory
figure('Name', 'End-Effector Trajectory');
plot3(ee_positions(:,1), ee_positions(:,2), ee_positions(:,3), 'b-', 'LineWidth', 1.5);
hold on;
plot3(ee_positions(1,1), ee_positions(1,2), ee_positions(1,3), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot3(ee_positions(end,1), ee_positions(end,2), ee_positions(end,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% Mark the time when force is applied
force_start_idx = find(tspan >= F_ext_start, 1);
force_end_idx = find(tspan >= F_ext_end, 1);
plot3(ee_positions(force_start_idx,1), ee_positions(force_start_idx,2), ee_positions(force_start_idx,3), 'ms', 'MarkerSize', 10, 'LineWidth', 2);
plot3(ee_positions(force_end_idx,1), ee_positions(force_end_idx,2), ee_positions(force_end_idx,3), 'cs', 'MarkerSize', 10, 'LineWidth', 2);

grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('End-Effector Trajectory');
legend('Trajectory', 'Start', 'End', 'Force Applied', 'Force Removed');
view(-37.5, 30);
axis equal;

% Plot 5: Momentum Residual
figure('Name', 'Momentum Residual');
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, r_history(:, j), 'b-', 'LineWidth', 1.5);
    
    ylabel(['Joint ' num2str(j) ' Residual']);
    if j == 1
        title('Momentum Observer Residual');
    end
    if j == 3
        xlabel('Time (s)');
    end
    grid on;
end

% Plot 6: Control Torques (tau_ctrl)
figure('Name', 'Control Torques');
for j = 1:3
    subplot(3, 1, j);
    plot(tspan, tau_history(:, j), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Plot the time window when external force is applied
    yline(0, 'k--');
    xline(F_ext_start, 'g--', 'Force Applied');
    xline(F_ext_end, 'r--', 'Force Removed');
    
    ylabel(['Joint ' num2str(j) ' (Nm)']);
    if j == 1
        title('Control Torques (PID Controller)');
    end
    if j == 3
        xlabel('Time (s)');
    end
    grid on;
end

%% Define the system's equations of motion
function dqdt = manipulator_dynamics(t, state, robot, tau)
    q = state(1:3);
    q_dot = state(4:6);

    M = massMatrix(robot, q);
    C = velocityProduct(robot, q, q_dot);
    G = gravityTorque(robot, q);

    q_ddot = M\(tau - C - G);

    dqdt = [q_dot; q_ddot];
end
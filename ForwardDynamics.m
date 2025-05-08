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

% 2. Define Initial Conditions
q0 = [0; 0; 0]; % Initial joint angles
q_dot0 = [0; 0; 0]; % Initial joint velocities

% 3. Define Simulation Parameters
t_start = 0;
t_end = 0.1;
dt = 0.01;
tspan = t_start:dt:t_end;

% 4. Define Control Torques (Example: Constant torques)
tau = [0;0;0]; % Torques for each joint

disp('Gravity torque at initial position:');
G_initial = gravityTorque(robot, q0);
disp(G_initial);

% 5. Forward Dynamics Computation

[t, state] = ode45(@(t, state) manipulator_dynamics(t, state, robot, tau), tspan, [q0; q_dot0]);

% 6. Extract Results
q = state(:, 1:3);
q_dot = state(:, 4:6);

% 7. Visualize Results
figure;
show(robot, q(1,:)');

for i = 1:length(t)
  pause(0.01);
  show(robot, q(i,:)');
  drawnow;
end

% (Optional) Use your forward kinematics to get end-effector positions
% For example:
% [x, y, z] = ForwardKinematics(q(i,1), q(i,2), q(i,3)); % Assuming your ForwardKinematics function takes joint angles as input

% Create an improved joint position plot
figure;
hold on;
plot(t, q(:,1), 'r-', 'LineWidth', 2);  % Joint 1 in red with thicker line
plot(t, q(:,2), 'g-', 'LineWidth', 1.5); % Joint 2 in green
plot(t, q(:,3), 'b-', 'LineWidth', 1.5); % Joint 3 in blue
grid on;
title('Joint Positions Over Time');
xlabel('Time (s)');
ylabel('Joint Angles (rad)');
legend('Joint 1', 'Joint 2', 'Joint 3');
xlim([t_start t_end]);
% Add a horizontal line at y=0 for reference
yline(0, 'k--', 'Zero Position');
hold off;

% Define the system's equations of motion as a function (moved to the end)
function dqdt = manipulator_dynamics(t, state, robot, tau)
  q = state(1:3);
  q_dot = state(4:6);

  M = massMatrix(robot, q);
  C = velocityProduct(robot, q, q_dot);  % This already accounts for velocity
  G = gravityTorque(robot, q);

  q_ddot = M\(tau - C - G);  % Remove the element-wise multiplication

  dqdt = [q_dot; q_ddot];
end
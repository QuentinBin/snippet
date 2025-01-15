'''
Description: 基于李雅普诺夫的目标跟踪demo
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-20 12:41:48
LastEditTime: 2025-01-14 16:18:10
'''
import numpy as np
from matplotlib import pyplot as plt
class RobotDact:
    def __init__(self):
        self.S = 0.04          # Reference surface area (m^2)
        self.CD = 0.2         # Drag force coefficient
        self.KD = 44.6e-4      # Drag moment coefficient (Kg·m^2)
        self.CL = 3.41         # Lift force coefficient
        self.mb = 4            # Mass of the body (Kg)
        self.J = 3.1e-3        # Inertia of the body (Kg·m^2)
        self.L = 0.1          # Length of the tail (m)
        self.rho = 1000        # Density of water (Kg/m^3)
        self.m = 1.9625        # The virtual mass per unit tail length (Kg/m)
        self.c = 0.2           # Distance from the tail to the center of the mass (m)
        self.kf = 0.918        # Average scaling force coefficient
      
        # Calculate coefficients c1, c2, c3, c4
        self.c1 = self.m / (2 * self.mb ) * self.L**2
        self.c2 = self.L**2 / (2 * self.J) * self.m * self.c
        self.c3 = self.KD / self.J
        self.c4 = self.L**3 / (3 * self.J) * self.m
        
        self.c1_bar = self.m / (4 * self.mb ) * self.L**2
        self.c2_bar = self.L**2 / (4 * self.J) * self.m * self.c
        
        # control params
        self.omega_alpha = 2 * np.pi # rad/s
        self.alpha_A = 30
        self.k1 = 0.002
        self.kr = 200


    def fu(self, u, v):
        """Calculate the u component of the force"""
        velocity_magnitude = np.sqrt(u**2 + v**2)
        return -0.5 / self.mb * self.rho * self.S * self.CD * u * velocity_magnitude + 0.5 / self.mb * self.rho * self.S * self.CL * v * velocity_magnitude * np.arctan(v / (u+1e-5))
    
    def fv(self, u, v):
        """Calculate the v component of the force"""
        velocity_magnitude = np.sqrt(u**2 + v**2)
        return -0.5 / self.mb * self.rho * self.S * self.CD * v * velocity_magnitude - 0.5 / self.mb * self.rho * self.S * self.CL * u * velocity_magnitude * np.arctan(v / (u+1e-5))
    
    def dynamics(self, u, v, alpha_deg, alpha_ddot_deg, omega):
        """Compute the dynamics based on the given system of equations"""
        
        # Compute forces from fu and fv
        f_u = self.fu(u, v)
        f_v = self.fv(u, v)
        alpha = alpha_deg * np.pi/180
        alpha_ddot = alpha_ddot_deg * (np.pi / 180)**2
        # Dynamics equations
        u_dot = v * omega + f_u - self.c1 * alpha_ddot * np.sin(alpha)
        v_dot = -u * omega + f_v + self.c1 * alpha_ddot * np.cos(alpha)
        omega_dot = -self.c3 * omega**2 * np.sign(omega) - self.c2 * alpha_ddot * np.cos(alpha) - self.c4 * self.m * alpha_ddot
        
        return u_dot, v_dot, omega_dot
    
    def kinematics(self, u, v, psi, omega):
        """Compute the kinematics based on the system of equations"""
        
        # Kinematic equations
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        psi_dot = omega
        
        return x_dot, y_dot, psi_dot

    def tail_deflection(self, alpha_0, t):
        """Calculate the tail deflection angle as a function of time"""
        alpha = self.alpha_A * np.sin(self.omega_alpha * t) + alpha_0
        alpha_dot = self.omega_alpha * self.alpha_A * np.cos(self.omega_alpha * t)
        alpha_ddot = -self.omega_alpha**2 * self.alpha_A * np.sin(self.omega_alpha * t)
        return alpha, alpha_ddot
    
    def target_tracking_dynamics(self, x, y, xs, ys, u, v, psi, omega):
        """Compute the dynamics for target tracking (r, θ)"""
        
        # Compute the relative coordinates to the target
        xe = xs - x
        ye = ys - y
        
        # Compute the relative angle φ
        phi = np.arctan2(ye, xe)
        
        # Compute the relative distance r and angle θ
        r = np.sqrt(xe**2 + ye**2)
        theta = psi - phi
        
        # Dynamics equations for r and θ
        r_dot = -u * np.cos(theta) + v * np.sin(theta)
        theta_dot = (u / r) * np.sin(theta) + (v / r) * np.cos(theta) + omega
        
        return r, theta, r_dot, theta_dot
    
    def compute_I0(self, u, v, omega, theta):
        """Compute I0 term in the discriminant"""
        # Calculate each part of I0
        f_u = self.fu(u, v)
        f_v = self.fv(u, v)
        
        term1 = (self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.sin(theta))**2
        term2 = -2 * self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta)
        
        part3 = -np.cos(theta) * (v * omega + f_u) + np.sin(theta) * (-u * omega + f_v) + (u * np.sin(theta) + v * np.cos(theta)) * omega
        
        return term1 + term2 * part3
    
    def compute_I1(self, u, v, theta):
        """Compute I1 term in the discriminant"""
        return -2 * self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta) * (self.k1 + self.kr) * (-u * np.cos(theta) + v * np.sin(theta))
    
    def compute_I2(self, theta, r):
        """Compute I2 term in the discriminant"""
        return -2 * self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta) * self.k1 * self.kr * r
    
    def compute_I3(self, u, v, theta, r):
        """Compute I3 term in the discriminant"""
        return -2 * self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta) * (1 / r) * (u * np.sin(theta) + v * np.cos(theta))**2
    
    def compute_discriminant(self, u, v, omega, theta, r):
        """Compute the discriminant B^2 - 4AC"""
        I0 = self.compute_I0(u, v, omega, theta)  
        I1 = self.compute_I1(u, v, theta)
        I2 = self.compute_I2(theta, r)
        I3 = self.compute_I3(u, v, theta, r)
        
        discriminant = (I0 + I1 + I2 + I3)
        return discriminant

    def compute_alpha0(self, u, v, omega, theta, r):
        """Compute the tail fin offset alpha0"""
        discriminant = self.compute_discriminant(u, v, omega, theta, r)
        
        f_u = self.fu(u, v)
        f_v = self.fv(u, v)

        if discriminant >= 0:
            B = self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta)
            A = 0.5 * self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta)
            C = self.c1_bar * self.kf * self.omega_alpha**2 * self.alpha_A**2 * np.cos(theta) * (1-1/8*self.alpha_A**2) \
                + (self.k1 + self.kr) * (-u * np.cos(theta) + v * np.sin(theta)) + self.k1*self.kr*r \
                - np.cos(theta) * (v*omega + f_u) + np.sin(theta) * (-u*omega + f_v) \
                + (u * np.sin(theta) + v * np.cos(theta)) * (u / r) * np.sin(theta) + (v / r) * np.cos(theta) + omega
            
            if np.abs(A) > 1e-4:
                alpha0 = (-B + np.sqrt(discriminant)) / (2 * A)
            else:
                alpha0 = -C / B
            return alpha0
        else:
            raise ValueError("Discriminant is negative, no real solution for alpha0.")
        

    def update_position(self, x, y, psi, x_dot, y_dot, psi_dot, dt):
        """Update position based on kinematics"""
        x_new = x + x_dot * dt
        y_new = y + y_dot * dt
        psi_new = psi + psi_dot * dt
        return x_new, y_new, psi_new
    

# Initial position and target
controller = RobotDact()
x, y, psi = 0.0, 0.0, np.pi/2  # Initial position (0, 0) and heading 0
u, v, omega = 0.2, 0, 0
target_x, target_y = 4.0, 3.0  # Target position (5, 5)



# Setup for real-time plotting
fig, ax = plt.subplots()
ax.set_xlim(-1, 6)  # Set x-axis limits
ax.set_ylim(-1, 6)  # Set y-axis limits
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Robot Trajectory')

robot_path, = ax.plot([], [], 'bo-', markersize=4, label='Robot Path')
robot_position, = ax.plot([], [], 'ro', markersize=6, label='Current Position')
ax.legend()

# Track positions for plotting
robot_positions = []

# Simulation loop
dt = 0.1  # Time step
x_dot, y_dot, psi_dot = controller.kinematics(u, v, psi, omega)
r, theta, r_dot, theta_dot = controller.target_tracking_dynamics(x, y, target_x, target_y, u, v, psi, omega)  
for t_num in range(int(1000/dt)):
    # # control
    if np.abs(theta) > np.pi/4:
        alpha0 =  60 * np.sign(theta)
        # alpha0 = controller.compute_alpha0(u,v,omega,theta,r)
    else:
        alpha0 = controller.compute_alpha0(u,v,omega,theta,r)
    alpha, alpha_ddot = controller.tail_deflection(alpha0, t_num*dt)
    u_dot, v_dot, omega_dot = controller.dynamics(u,v,alpha,alpha_ddot,omega)

    # update velocity
    u += u_dot * dt
    v += v_dot * dt
    omega += omega_dot * dt
    x_dot, y_dot, psi_dot = controller.kinematics(u, v, psi, omega)
    
    # update position
    x, y, psi = controller.update_position(x, y, psi, x_dot, y_dot, psi_dot, dt)
    r, theta, r_dot, theta_dot = controller.target_tracking_dynamics(x, y, target_x, target_y, u, v, psi, omega)

    # Append current position for plotting
    robot_positions.append((x, y))

    # Clear the current figure
    # plt.clf()
    
    # Plot the current path
    x_vals = [pos[0] for pos in robot_positions]
    y_vals = [pos[1] for pos in robot_positions]
    # plt.plot(x_vals, y_vals, 'bo-', markersize=4, label='Robot Path')
    
    # # Plot the current position
    # plt.plot(x, y, 'ro', markersize=6, label='Current Position')
    
    # # Plot the target position
    # plt.plot(target_x, target_y, 'gx', markersize=8, label='Target Position')
    
    # # Set plot properties
    # plt.xlim(-1, 6)
    # plt.ylim(-1, 6)
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.title('Robot Trajectory')
    # plt.legend()
    
    # Show plot and pause for a short time
    # plt.pause(dt/10)
    
    # Print position and status
    print(f"Time: {t_num*dt:.2f}s, Position: ({x:.2f}, {y:.2f}), Heading: {psi:.2f}, Distance to target: {r:.2f}")

    # Stop if the robot reaches the target
    if r <= 0.2:
        print("Target reached!")
        break


plt.clf()
    
# Plot the current path
x_vals = [pos[0] for pos in robot_positions]
y_vals = [pos[1] for pos in robot_positions]
plt.plot(x_vals, y_vals, 'bo-', markersize=4, label='Robot Path')

# Plot the current position
plt.plot(x, y, 'ro', markersize=6, label='Current Position')

# Plot the target position
plt.plot(target_x, target_y, 'gx', markersize=8, label='Target Position')

# Set plot properties
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Trajectory')
plt.legend()
plt.show()
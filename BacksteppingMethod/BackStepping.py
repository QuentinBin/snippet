'''
Description: 基于李雅普诺夫的目标跟踪demo
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-20 12:41:48
LastEditTime: 2025-03-12 13:02:30
'''
import numpy as np
from matplotlib import pyplot as plt
class RobotDact:
    def __init__(self):
        self.m = 17.446
        self.J = 0.446
        self.m_ax = 0
        self.m_ay = 5.559


        self.C_dx = 0.1445 + 0.1309 + 0.1070
        self.C_dy = 0.5 + 0.4098 + 0.2273
        self.K_d = .6
        self.rho = 1000
        
        # control params
        self.pec_amp_foward = 0
        self.pec_amp_turn = 0
        self.tail_bias = 0
        # control hyper_params
        self.pec_freq = 1.0
        self.tail_freq = 1.0
        self.tail_amp = 60
        self.k_pec1 = 0
        self.k_pec2 = 0

        # controller params
        self.k1 = 0.002
        self.k2 = 0.002
        self.kr = 200
        self.kp = 200
    
    def compute_pec_force(pec_freq, pec_amp_foward, pec_amp_turn):
        coefficients = np.array([
            [0.9262, -1.6480,  1.5960,  0.6419,  0.0442, -1.5290],
            [0.1016,  1.5580, -2.0650,  1.4700, -0.1958,  3.0170],
            [0.0535,  0.7496,  1.3190,  0.3941,  0.1191, -0.7917]
        ])

        pec_left_amp = pec_amp_foward - pec_amp_turn
        pec_right_amp = pec_amp_foward + pec_amp_turn

        pec_left_force = 0
        pec_right_force = 0
        for row in coefficients:
            a_n, b_n, c_n, d_n, e_n, f_n = row
            term_l = (a_n * pec_freq**3 + b_n * pec_freq**2 + c_n * pec_freq + d_n) * (e_n * pec_left_amp + f_n)
            pec_left_force += term_l

            term_r = (a_n * pec_freq**3 + b_n * pec_freq**2 + c_n * pec_freq + d_n) * (e_n * pec_right_amp + f_n)
            pec_right_force += term_r
        pec_torque = (pec_right_force - pec_right_force) * 0.15

        return pec_left_force, pec_right_force, pec_torque
    
    def compute_tail_force(tail_freq, tail_bias, amp=60):
        tail_force = 0.02 * tail_freq**2 * (1 - 4.74 * (tail_bias/180*3.14)**2)
        return tail_force


    def fup(self, u, v, w):
        term1 = (self.m - self.m_ay)/(self.m-self.m_ax) * v * w
        term2 = -0.5 * self.rho * self.C_dx * u**2 / (self.m-self.m_ax)
        term3 = 2* self.k_pec2 /(self.m-self.m_ax)
        return term1+term2+term3
    
    def fvp(self, u, v, w):
        term1 = -(self.m - self.m_ax)/(self.m-self.m_ay) * v * w
        term2 = -0.5 * self.rho * self.C_dy * v**2 / (self.m-self.m_ay)
        return term1+term2
    
    def fvp(self, u, v, w):
        term1 = (self.m_ay - self.m_ax)/(self.J) * u * v
        term2 = -self.K_d*np.sign(w)*w**2 /(self.J)
        return term1+term2
    
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
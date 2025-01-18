## Author: Joao Moura
## Contact: jpousad@ed.ac.uk
## Date: 15/12/2020
## -------------------------------------------------------------------
## Description:
## 
## Functions for outputting different nominal trajectories
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## Import libraries
## -------------------------------------------------------------------
import numpy as np
import casadi as cs

## Generate Nominal Trajectory (line)
def generate_traj_line(x_f, y_f, N, N_MPC):
    x_nom = np.linspace(0.0, x_f, N)
    y_nom = np.linspace(0.0, y_f, N)
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_f+x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_f+y_nom[1:N_MPC+1]), axis=0)
# def generate_traj_sine(x_f, y_f, A, N, N_MPC):
#     x_nom = np.linspace(0.0, x_f, N)
#     y_nom = A * np.sin(np.linspace(0.0, 2 * np.pi, N))
#     # return x_nom, y_nom
#     return np.concatenate((x_nom, x_f+x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_f+y_nom[1:N_MPC+1]), axis=0)
def generate_traj_sine(x_f, y_f, A, N, N_MPC):
    x_nom = A * np.sin(np.linspace(0.0, 2 * np.pi, N))
    y_nom = np.linspace(0.0, y_f, N)
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_f+x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_f+y_nom[1:N_MPC+1]), axis=0)
def generate_traj_circle(theta_i, theta_f, radious, N, N_MPC):
    s = np.linspace(theta_i, theta_f, N)
    # s = s[::-1] # reverse the order of the angle, for debugging purposes
    x_nom = radious*np.cos(s)
    y_nom = radious*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_dubins_curve(theta_i, dtheta, center_x, center_y, radious, N, N_MPC):
    s = np.linspace(theta_i, theta_i + dtheta, N)
    x_nom = center_x+radious*np.cos(s)
    y_nom = center_y+radious*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_traj_ellipse(theta_i, theta_f, radious_x, radious_y, N, N_MPC):
    s = np.linspace(theta_i, theta_f, N)
    x_nom = radious_x*np.cos(s)
    y_nom = radious_y*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def generate_traj_eight(side_lenght, N, N_MPC):
    s = np.linspace(0.0, 2*np.pi, N)
    x_nom = side_lenght*np.sin(s)
    y_nom = side_lenght*np.sin(s)*np.cos(s)
    # return x_nom, y_nom
    return np.concatenate((x_nom, x_nom[1:N_MPC+1]), axis=0), np.concatenate((y_nom, y_nom[1:N_MPC+1]), axis=0)
def compute_nomState_from_nomTraj(x_data, y_data, theta_0, dt):
    # assign two first state trajectories
    x0_nom = x_data
    x1_nom = y_data
    # compute diff for planar traj
    Dx0_nom = np.diff(x0_nom)
    Dx1_nom = np.diff(x1_nom)
    # compute traj angle 
    ND = len(Dx0_nom)
    x2_nom = np.empty(ND)
    theta = 0.0
    for i in range(ND):
        # align +x axis with the forwarding direction
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, s), (-s, c)))
        Dx_new = R.dot(np.array((Dx0_nom[i],Dx1_nom[i])))
        # print(Dx_new)
        theta += np.arctan2(Dx_new[1], Dx_new[0])
        x2_nom[i] = theta
    x2_nom = np.append(x2_nom, x2_nom[-1])
    x2_nom -= x2_nom[0] # initial angle at zero
    Dx2_nom = np.diff(x2_nom)
    # specify angle of the pusher relative to slider
    x3_nom = np.zeros_like(x0_nom)
    Dx3_nom = np.diff(x3_nom)
    # stack state and derivative of state
    # x_nom = np.vstack((x0_nom, x1_nom, x2_nom, x3_nom))
    x_nom = cs.horzcat(x0_nom, x1_nom, x2_nom, x3_nom).T
    dx_nom = cs.horzcat(Dx0_nom, Dx1_nom, Dx2_nom, Dx3_nom).T/dt
    return x_nom, dx_nom

if __name__ == "__main__":
    T = 30  # time of the simulation is seconds
    freq = 20  # number of increments per second
    N_MPC = 30  # time horizon for the MPC controller
    N = int(T*freq)  # total number of iterations
    dt = 1.0/freq  # sampling time
    x_init_val = [0.0, 0.0, 0.0, 0.0]
    x0_nom, x1_nom = generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
    X_nom_val, _ = compute_nomState_from_nomTraj(x0_nom, x1_nom, x_init_val[2], dt)
    thetas = np.array(X_nom_val[2,:]).reshape(-1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(thetas)
    plt.show()

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import dyn_sliding_pack
from dyn_sliding_pack import math_utils


__kCoordAxisLength = 0.1
__KMaxForceArrowLength = 0.3
__KMaxVelArrowLength = 0.3

def get_arrow_length(f, f_max, l_max):
    return l_max * (f / f_max)

def skew(v):
    """
        3x3 skew-symmetric matrix
        v: angle
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def dir_cosine(q):
    """
        3x3 rotation matrix
        q: quaternion in [w, x, y, z] order
    """
    return np.array([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
        [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
        [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])

def omega(w):
    """
        4x4 skew-symmetric matrix
        w: 3x vector of angular velocity
    """
    return np.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])

def enum_board_corner_coord(m, xi):
    Ri = dir_cosine(xi[3:7])
    x_axis_i, y_axis_i = Ri[:, 0], Ri[:, 1]
    board_coords = []
    board_coords.append(xi[0:3] + (m.lx/2) * x_axis_i + (m.ly/2) * y_axis_i)
    board_coords.append(xi[0:3] - (m.lx/2) * x_axis_i + (m.ly/2) * y_axis_i)
    board_coords.append(xi[0:3] - (m.lx/2) * x_axis_i - (m.ly/2) * y_axis_i)
    board_coords.append(xi[0:3] + (m.lx/2) * x_axis_i - (m.ly/2) * y_axis_i)
    return np.stack(board_coords)

def add_axis_of_coordinates(ax, x, plot_every):
    N = x.shape[0]

    arrow_start = []
    arrow_incre = []

    for i in range(N):
        if (i % plot_every == 0) or (i == N-1):
            xi = x[i, :]
            Ri = dir_cosine(xi[3:7])
            arrow_start += ([xi[0:3].tolist()]*3)
            arrow_incre += ([Ri[:,0].tolist()])
            arrow_incre += ([Ri[:,1].tolist()])
            arrow_incre += ([Ri[:,2].tolist()])

    arrow_start = np.stack(arrow_start)
    arrow_incre = np.stack(arrow_incre)

    # x-axis
    X, Y, Z = arrow_start[0::3, 0].tolist(), arrow_start[0::3, 1].tolist(), arrow_start[0::3, 2].tolist()
    U, V, W = arrow_incre[0::3, 0].tolist(), arrow_incre[0::3, 1].tolist(), arrow_incre[0::3, 2].tolist()
    ax.quiver(X, Y, Z, U, V, W, length=__kCoordAxisLength, normalize=True, color='red')

    # y-axis
    X, Y, Z = arrow_start[1::3, 0].tolist(), arrow_start[1::3, 1].tolist(), arrow_start[1::3, 2].tolist()
    U, V, W = arrow_incre[1::3, 0].tolist(), arrow_incre[1::3, 1].tolist(), arrow_incre[1::3, 2].tolist()
    ax.quiver(X, Y, Z, U, V, W, length=__kCoordAxisLength, normalize=True, color='green')

    # z-axis
    X, Y, Z = arrow_start[2::3, 0].tolist(), arrow_start[2::3, 1].tolist(), arrow_start[2::3, 2].tolist()
    U, V, W = arrow_incre[2::3, 0].tolist(), arrow_incre[2::3, 1].tolist(), arrow_incre[2::3, 2].tolist()
    ax.quiver(X, Y, Z, U, V, W, length=__kCoordAxisLength, normalize=True, color='blue')

def plot_knot_points(m, x, u, plot_every=3):
    """
        Plot the TO knot points, frictional forces and inertia forces
        m: the phisics model
        x: the optimal states
        u: the optimal control effort
    """
    N = x.shape[0]
    cmap = mpl.colormaps['tab10']

    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    for i in range(N):
        if (i % plot_every == 0) or (i == N-1):
            xi = x[i, :]
            ui = u[i, :]

            # plot the pusher board
            boardi = enum_board_corner_coord(m, xi)
            vertsi = [list(zip(boardi[:,0].tolist(), boardi[:,1].tolist(), boardi[:,2].tolist()))]
            ax.add_collection3d(Poly3DCollection(vertsi, facecolor=cmap(7), alpha=0.5, edgecolor=(1,1,1,0), linewidth=0))

            # plot the ball
            Ri = dir_cosine(xi[3:7])
            balli = xi[0:3] + xi[13] * Ri[:, 0] + xi[14] * Ri[:, 1]  # + m.r * Ri[:, 2]
            ax.scatter(balli[0], balli[1], balli[2], s=100, marker='o', facecolor=cmap(2), edgecolor=(0,0,0,0))

            # plot the gravity force
            f_grav_g = np.array([0., 0., -m.m * m.g])
            ax.quiver(balli[0], balli[1], balli[2], f_grav_g[0], f_grav_g[1], f_grav_g[2], \
                      length=__KMaxForceArrowLength, normalize=True, color=cmap(4), alpha=0.5, \
                      arrow_length_ratio=0.1)
            
            # plot the board contact force
            f_fric_b = Ri.dot(ui[0:3])
            f_fric_b = get_arrow_length(f_fric_b, m.m*m.g, __KMaxForceArrowLength)
            ax.quiver(balli[0], balli[1], balli[2], f_fric_b[0], f_fric_b[1], f_fric_b[2], \
                      length=np.linalg.norm(f_fric_b), normalize=True, color=cmap(1), alpha=0.5, \
                      arrow_length_ratio=0.1)
            
            # plot the inertia force
            f_iner_b = Ri.dot(-m.m*skew(xi[m.w_b_idx])@skew(xi[m.w_b_idx])@np.append(xi[m.r_m_idx], 0.0) \
                              -2*m.m*skew(xi[m.w_b_idx])@np.append(xi[m.v_m_idx], 0.0) \
                              -m.m*skew(ui[m.dw_b_idx])@np.append(xi[m.r_m_idx], 0.0) \
                              -m.m*Ri.T@ui[m.a_b_idx])
            f_iner_b = get_arrow_length(f_iner_b, m.m*m.g, __KMaxForceArrowLength)
            ax.quiver(balli[0], balli[1], balli[2], f_iner_b[0], f_iner_b[1], f_iner_b[2], \
                      length=np.linalg.norm(f_iner_b), normalize=True, color=cmap(9), alpha=0.5, \
                      arrow_length_ratio=0.1)

    # plot coordinates
    add_axis_of_coordinates(ax, x, plot_every)

    plt.show()

def plot_statistics(m, x, u, tf):
    """
        tf: final time
    """
    N = x.shape[0]
    t_grid = np.linspace(0, tf, N)

    x_name = ['x_board', 'y_board', 'z_board', 'q0_board', 'q1_board', \
              'q2_board', 'q3_board', 'dx_board', 'dy_board', 'dz_board', \
              'omega0_board', 'omega1_board', 'omega2_board', \
              'x_mass', 'y_mass', 'dx_mass', 'dy_mass']
    
    u_name = ['f_tan0', 'f_tan1', 'f_norm', \
              'd2x_board', 'd2y_board', 'd2z_board', \
              'beta0_board', 'beta1_board', 'beta2_board', 'k']
    
    fig = plt.figure(figsize=(14, 12))
    fig.canvas.manager.set_window_title('states')

    # ------------------------------------

    # states
    for i_row in range(5):
        for i_col in range(4):
            i_item = (i_row)*4+i_col
            if i_item >= len(x_name):
                break
            ax = fig.add_subplot(5, 4, i_item+1)
            if m.lbx[i_item] > -np.inf:
                ax.axhline(m.lbx[i_item], color='red', linestyle='--', linewidth=2)
            if m.ubx[i_item] < np.inf:
                ax.axhline(m.ubx[i_item], color='red', linestyle='--', linewidth=2)
            ax.plot(t_grid, x[:, i_item], label=x_name[i_item], marker='o', markersize=4, linewidth=2)
            ax.grid('on'); ax.legend()

    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title('inputs')

    # control efforts
    for i_row in range(2):
        for i_col in range(5):
            i_item = (i_row)*5+i_col
            ax = fig.add_subplot(2, 5, i_item+1)
            if m.lbu[i_item] > -np.inf:
                ax.axhline(m.lbu[i_item], color='red', linestyle='--', linewidth=2)
            if m.ubu[i_item] < np.inf:
                ax.axhline(m.ubu[i_item], color='red', linestyle='--', linewidth=2)
            ax.plot(t_grid, u[:, i_item], label=u_name[i_item], marker='o', markersize=4, linewidth=2)
            if i_item == 2:
                ax.plot(t_grid, np.sqrt(u[:, 0]**2+u[:, 1]**2)/m.mu_p, label='||ft||/mu', marker='o', markersize=4, linewidth=2)
            ax.grid('on'); ax.legend()

    # ------------------------------------

    # constraints
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title('constraints')

    eq_cst = np.empty((m.n_eq, N))
    ineq_cst = np.empty((2, N))
    for i in range(N):
        eq_cst[:, i] = m.eq_func(x[i, :], u[i, :]).flatten()
        ineq_cst[0, i] = np.linalg.norm(u[i, 0:2]) - m.mu_p * u[i, 2]
        ineq_cst[1, i] = u[i, 9] + m.eps

    for i in range(4):
        ax = fig.add_subplot(2, 4, i+1)
        ax.axhline(0.0, color='red', linestyle='--', linewidth=2)
        ax.plot(t_grid, eq_cst[i, :], label='eq{0}'.format(i), marker='o', markersize=4, linewidth=2)
        ax.grid('on'); ax.legend()
    
    ax = fig.add_subplot(2, 4, 5)
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2)
    ax.plot(t_grid, ineq_cst[0, :], label='ineq0', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    ax = fig.add_subplot(2, 4, 6)
    ax.axhline(0.0, color='red', linestyle='--', linewidth=2)
    ax.plot(t_grid, ineq_cst[1, :], label='ineq1', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    # ------------------------------------

    # complementarity
    fig = plt.figure(figsize=(8, 3))
    fig.canvas.manager.set_window_title('complementarity')

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(t_grid, m.mu_p*u[:, 2]-np.linalg.norm(u[:, 0:2], axis=1), label='cmpl0_0', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, np.linalg.norm(x[:, 15:17], axis=1), label='cmpl0_1', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()
    
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(t_grid, u[:, 0]-u[:, 9]*x[:, 15], label='cmpl1_0', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, np.linalg.norm(x[:, 15:17], axis=1), label='cmpl1_1', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t_grid, u[:, 1]-u[:, 9]*x[:, 16], label='cmpl2_0', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, np.linalg.norm(x[:, 15:17], axis=1), label='cmpl2_1', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    plt.show()


def plot(m, x, u, tf, J, L):
    # plot_knot_points(m, x[-1].transpose(1, 0), u[-1].transpose(1, 0), plot_every=2)
    plot_statistics(m, x[-1].transpose(1, 0), u[-1].transpose(1, 0), tf[-1])

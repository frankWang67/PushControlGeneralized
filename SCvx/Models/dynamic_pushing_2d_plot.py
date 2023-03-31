import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation, patches, transforms
import numpy as np

import dyn_sliding_pack
from dyn_sliding_pack import math_utils


__KMaxForceArrowLength = 0.3
__KMaxVelArrowLength = 0.3

def get_arrow_length(f, f_max, l_max):
    return l_max * (np.linalg.norm(f) / f_max)

class ResultAnimation():
    def __init__(self, m) -> None:
        self.h_board = m.h
        self.l_board = m.l
        self.r_ball = m.r

    def set_patches_for_animation(self, ax, x0):
        x_board0 = x0[0:3]

        R0 = math_utils.rotation_matrix(x_board0[2])
        disp = -R0.dot(np.array([self.l_board/2, self.h_board/2]))
        self.board = patches.Rectangle(x_board0[0:2]+disp[0:2], self.l_board, self.h_board, angle=x_board0[2], color='#1f77b4')

        theta_board0 = x_board0[2]
        unit_x_board0, unit_y_board0 = np.array([np.cos(theta_board0), np.sin(theta_board0)]), \
                                    np.array([-np.sin(theta_board0), np.cos(theta_board0)])
        x_ball0 = x_board0[0:2] + (self.h_board/2+self.r_ball) * unit_y_board0 + x0[3] * unit_x_board0

        self.ball = patches.Circle(x_ball0, radius=self.r_ball, color='black')
        
        ax.add_patch(self.board)
        ax.add_patch(self.ball)

    def update_for_animation(self, i, ax, x):
        xi = x[i, :].flatten()
        x_boardi = xi[0:3]

        R0 = math_utils.rotation_matrix(x_boardi[2])
        disp = -R0.dot(np.array([self.l_board/2, self.h_board/2]))

        ci = x_boardi[0:2] + disp
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], x_boardi[2])
        self.board.set_transform(trans_ax+trans_i)
        self.board.set_xy([ci[0], ci[1]])

        theta_board0 = x_boardi[2]
        unit_x_board0, unit_y_board0 = np.array([np.cos(theta_board0), np.sin(theta_board0)]), \
                                    np.array([-np.sin(theta_board0), np.cos(theta_board0)])
        x_ball0 = x_boardi[0:2] + (self.h_board/2+self.r_ball) * unit_y_board0 + xi[3] * unit_x_board0

        self.ball.set_center(x_ball0)
        
        return []

def plot_animation(m, x):
    """
        Plot the ball-and-board system as an animation
        m: the phisics model
        x: the optimal states
    """
    # plot animation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.xlim([-0.1, 0.6])
    plt.ylim([-0.1, 0.6])
    animator = ResultAnimation(m)
    animator.set_patches_for_animation(ax, x[0, :].flatten())

    N = x.shape[0]

    anim = animation.FuncAnimation(
        fig,
        animator.update_for_animation,
        fargs=(ax, x),
        frames=N,
        interval=100,
        blit=True,
        repeat=False,
    )
    plt.show()

def plot_knot_points(m, x, u):
    """
        Plot the TO knot points, frictional forces and inertia forces
        m: the phisics model
        x: the optimal states
        u: the optimal control effort
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.xlim([-0.1, 0.6])
    plt.ylim([-0.1, 0.6])

    N = x.shape[0]
    cmap = mpl.colormaps['tab10']
    p_ball_all = np.empty((N, 2))

    for i in range(N):
        xi = x[i, :]
        ui = u[i, :]
        Ri = math_utils.rotation_matrix(xi[2])

        # plot the ball
        p_ball = xi[0:2] + Ri.dot(xi[3]*np.array([1, 0]))
        p_ball_all[i, :] = p_ball
        ax.scatter(p_ball[0], p_ball[1], marker='o', facecolor=cmap(2), edgecolor=(0,0,0,0), s=50)

        if i % 3 == 0:
            # plot the board as a line
            p_end0 = xi[0:2] + Ri.dot((m.l/2)*np.array([1, 0]))
            p_end1 = xi[0:2] + Ri.dot((-m.l/2)*np.array([1, 0]))
            ax.plot([p_end0[0], p_end1[0]], [p_end0[1], p_end1[1]], linewidth=2, color=cmap(5), alpha=0.5)

            # plot the inertia force
            f_iner_b = m.f_iner_b_func(xi, ui).flatten()
            f_iner_b = get_arrow_length(f_iner_b, m.f_max, __KMaxForceArrowLength)
            ax.arrow(p_ball[0], p_ball[1], f_iner_b[0], f_iner_b[1], width=0.005, facecolor=cmap(9), alpha=0.5, edgecolor=(1,1,1,0))

            # plot the ground frictional force
            f_fric_g = m.f_fric_g_func(xi, ui).flatten()
            f_fric_g = get_arrow_length(f_fric_g, m.f_max, __KMaxForceArrowLength)
            ax.arrow(p_ball[0], p_ball[1], f_fric_g[0], f_fric_g[1], width=0.005, facecolor=cmap(4), alpha=0.5, edgecolor=(1,1,1,0))

            # plot the ball's ground velocity
            v_mass_g = m.v_mass_g_func(xi, ui).flatten()
            v_mass_g = get_arrow_length(v_mass_g, m.v_max, __KMaxVelArrowLength)
            ax.arrow(p_ball[0], p_ball[1], v_mass_g[0], v_mass_g[1], width=0.005, facecolor=cmap(4), alpha=0.5, edgecolor=(1,1,1,0))

            # plot the board frictional force
            f_fric_b = m.f_fric_b_func(xi, ui).flatten()
            f_fric_b = get_arrow_length(f_fric_b, m.f_max, __KMaxForceArrowLength)
            ax.arrow(p_ball[0], p_ball[1], f_fric_b[0], f_fric_b[1], width=0.005, facecolor=cmap(1), alpha=0.5, edgecolor=(1,1,1,0))

    # plot the balls, connected with lines
    ax.plot(p_ball_all[:, 0], p_ball_all[:, 1], marker='o', color=cmap(2), linewidth=2, markersize=5, alpha=0.5)

    plt.show()

def plot_statistics(m, x, u, tf):
    """
        tf: final time
    """
    N = x.shape[0]
    t_grid = np.linspace(0, tf, N)

    fig = plt.figure(figsize=(10, 8))
    
    # state and control efforts
    for i_row in range(2):
        for i_col in range(4):
            i_item = (i_row)*4+i_col
            ax = fig.add_subplot(3, 4, i_item+1)
            ax.plot(t_grid, x[:, i_item], label='x{0}'.format(i_item), marker='o', markersize=4, linewidth=2)
            ax.grid('on'); ax.legend()

    ax = fig.add_subplot(3, 4, 9)
    ax.plot(t_grid, u[:, 0], label='f_n', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, u[:, 1], label='f_t', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    ax = fig.add_subplot(3, 4, 10)
    ax.plot(t_grid, u[:, 6], label='d2x0', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()
    
    ax = fig.add_subplot(3, 4, 11)
    ax.plot(t_grid, u[:, 7], label='d2x1', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    ax = fig.add_subplot(3, 4, 12)
    ax.plot(t_grid, u[:, 8], label='d2x2', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    # ------------------------------------

    # constraint violation
    eq_cstr = np.empty((N, m.n_eq))
    for i in range(N):
        xi = x[i, :]
        ui = u[i, :]
        eq_cstr[i, :] = m.eq_func(xi, ui).flatten()
    
    ineq_cstr = np.empty((N, m.n_ineq))
    for i in range(N):
        xi = x[i, :]
        ui = u[i, :]
        ineq_cstr[i, :] = m.ineq_func(xi, ui).flatten()

    fig = plt.figure(figsize=(8, 6))
    for i_row in range(2):
        for i_col in range(3):
            i_item = (i_row)*3+i_col
            if i_item < 5:
                ax = fig.add_subplot(2, 3, i_item+1)
                ax.plot(t_grid, eq_cstr[:, i_item], label='eq{0}'.format(i_item), marker='o', markersize=4, linewidth=2)
                ax.grid('on'); ax.legend()
            else:
                ax = fig.add_subplot(2, 3, i_item+1)
                ax.plot(t_grid, ineq_cstr[:, 0], label='ineq{0}'.format(i_item), marker='o', markersize=4, linewidth=2)
                ax.grid('on'); ax.legend()

    # ------------------------------------
    
    # complementary constraints
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t_grid, m.mu_p * u[:, 0] - u[:, 1], label='mu*fn-ft', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, u[:, 5], label='dx-', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(t_grid, m.mu_p * u[:, 0] + u[:, 1], label='mu*fn+ft', marker='o', markersize=4, linewidth=2)
    ax.plot(t_grid, u[:, 4], label='dx+', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    plt.show()

def plot_convergence(x, J, L):
    fig = plt.figure(figsize=(10, 4))
    
    N = J.shape[0]
    assert L.shape[0] == N-1

    # total cost
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(np.arange(N), J[:, 2], label='non-linear', marker='o', markersize=4, linewidth=2)
    ax.plot(np.arange(1, N), L[:, 2], label='linear', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    # dynamic violation
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(np.arange(N), J[:, 0], label='non-linear', marker='o', markersize=4, linewidth=2)
    ax.plot(np.arange(1, N), L[:, 0], label='linear', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    # constraint violation
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(np.arange(N), J[:, 1], label='non-linear', marker='o', markersize=4, linewidth=2)
    ax.plot(np.arange(1, N), L[:, 1], label='linear', marker='o', markersize=4, linewidth=2)
    ax.grid('on'); ax.legend()

    # solution convergence
    ax = fig.add_subplot(1, 4, 4)
    dx_norm = np.linalg.norm((x[:-1, ...] - x[-1, ...]).reshape(N-1, -1), axis=1)
    ax.plot(np.arange(N-1), dx_norm, marker='o', markersize=4, linewidth=2)
    ax.set_yscale('log')
    ax.grid('on'); ax.legend()

    plt.show()


def plot(m, x, u, tf, J, L):
    plot_convergence(x, J, L)
    plot_statistics(m, x[-1].transpose(1, 0), u[-1].transpose(1, 0), tf[-1])
    plot_knot_points(m, x[-1].transpose(1, 0), u[-1].transpose(1, 0))

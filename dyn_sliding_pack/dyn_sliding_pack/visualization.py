from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import transforms
import numpy as npy
import dyn_sliding_pack


def plot_variable_and_constraint(pbm:dyn_sliding_pack.params.DynPusherOptimizationConfig, sol, N, tf):
    nx = pbm.x.shape[0]
    nu = pbm.u.shape[0]
    nz = pbm.z.shape[0]
    np = pbm.p.shape[0]
    
    x_opt = sol['x'][:N*nx].toarray().reshape(N,nx)
    u_opt = sol['x'][N*nx:N*nx+N*nu].toarray().reshape(N,nu)
    z_opt = sol['x'][N*nx+N*nu:N*nx+N*nu+N*nz].toarray().reshape(N,nz)

    # plot variables
    t_grid = npy.linspace(0, tf, N)
    fig = plt.figure(figsize=(20, 10))
    n_subplot = max(nx, nu, nz)
    for i in range(nx):
        ax = fig.add_subplot(3, n_subplot, 1+i)
        ax.plot(t_grid, x_opt[:, i], label='x_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    for i in range(nu):
        ax = fig.add_subplot(3, n_subplot, n_subplot+1+i)
        ax.plot(t_grid, u_opt[:, i], label='u_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    for i in range(nz):
        ax = fig.add_subplot(3, n_subplot, n_subplot*2+1+i)
        ax.plot(t_grid, z_opt[:, i], label='z_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    # plot constraints
    dim_dyn = pbm.f.size_out('f')[0]
    dim_eq = pbm.h.size_out('h')[0]
    dim_ieq = pbm.g.size_out('g')[0]
    dyn_con = sol['g'][:(N-1)*dim_dyn].toarray().reshape(N-1,dim_dyn)
    eq_con = sol['g'][(N-1)*dim_dyn:(N-1)*dim_dyn+N*dim_eq].toarray().reshape(N,dim_eq)
    ieq_con = sol['g'][(N-1)*dim_dyn+N*dim_eq:(N-1)*dim_dyn+N*dim_eq+N*dim_ieq].toarray().reshape(N,dim_ieq)

    fig = plt.figure(figsize=(20, 10))
    n_subplot = max(dim_dyn, dim_eq, dim_ieq)
    for i in range(dim_dyn):
        ax = fig.add_subplot(3, n_subplot, 1+i)
        ax.plot(t_grid[:-1], dyn_con[:, i], label='dyn_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    for i in range(dim_eq):
        ax = fig.add_subplot(3, n_subplot, n_subplot+1+i)
        ax.plot(t_grid, eq_con[:, i], label='eq_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    for i in range(dim_ieq):
        ax = fig.add_subplot(3, n_subplot, n_subplot*2+1+i)
        ax.plot(t_grid, ieq_con[:, i], label='ieq_{0}'.format(i))
        ax.grid('on')
        ax.legend()

    # auxiliary variables for debug
    fric_ground = npy.empty((N, 2))
    for i in range(N):
        fric_ground[i, :] = pbm.aux_f_ground(x_opt[i, :], u_opt[i, :], [tf], z_opt[i, :]).toarray().flatten()
    
    inert_board = npy.empty((N, 2))
    for i in range(N):
        inert_board[i, :] = pbm.aux_f_inertia(x_opt[i, :], u_opt[i, :], [tf], z_opt[i, :]).toarray().flatten()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t_grid, fric_ground[:, 0], label='fric_g_x')
    ax.grid('on')
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t_grid, fric_ground[:, 1], label='fric_g_y')
    ax.grid('on')
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_grid, inert_board[:, 0], label='inert_b_x')
    ax.grid('on')
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t_grid, inert_board[:, 1], label='inert_b_y')
    ax.grid('on')
    ax.legend()

    plt.show()

class ResultAnimation():
    def __init__(self, pbm:dyn_sliding_pack.params.DynPusherProblem) -> None:
        self.h_board = pbm.system.h
        self.l_board = pbm.system.l
        self.r_ball = pbm.system.r

    def set_patches_for_animation(self, ax, x0):
        x_board0 = x0[0:3]

        R0 = dyn_sliding_pack.math_utils.rotation_matrix(x_board0[2])
        disp = -R0.dot(npy.array([self.l_board/2, self.h_board/2]))
        self.board = patches.Rectangle(x_board0[0:2]+disp[0:2], self.l_board, self.h_board, angle=x_board0[2], color='#1f77b4')

        theta_board0 = x_board0[2]
        unit_x_board0, unit_y_board0 = npy.array([npy.cos(theta_board0), npy.sin(theta_board0)]), \
                                    npy.array([-npy.sin(theta_board0), npy.cos(theta_board0)])
        x_ball0 = x_board0[0:2] + (self.h_board/2+self.r_ball) * unit_y_board0 + x0[3] * unit_x_board0

        self.ball = patches.Circle(x_ball0, radius=self.r_ball, color='black')
        
        ax.add_patch(self.board)
        ax.add_patch(self.ball)

    def update_for_animation(self, i, ax, x):
        xi = x[i, :].flatten()
        x_boardi = xi[0:3]

        R0 = dyn_sliding_pack.math_utils.rotation_matrix(x_boardi[2])
        disp = -R0.dot(npy.array([self.l_board/2, self.h_board/2]))

        ci = x_boardi[0:2] + disp
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], x_boardi[2])
        self.board.set_transform(trans_ax+trans_i)
        self.board.set_xy([ci[0], ci[1]])

        theta_board0 = x_boardi[2]
        unit_x_board0, unit_y_board0 = npy.array([npy.cos(theta_board0), npy.sin(theta_board0)]), \
                                    npy.array([-npy.sin(theta_board0), npy.cos(theta_board0)])
        x_ball0 = x_boardi[0:2] + (self.h_board/2+self.r_ball) * unit_y_board0 + xi[3] * unit_x_board0

        self.ball.set_center(x_ball0)
        
        return []

import os
r3t_root_dir = os.environ.get("R3T_HOME")

import warnings
warnings.filterwarnings('ignore')

#  import libraries
#  -------------------------------------------------------------------
import time
import numpy as np
import casadi as cs
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import patches, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.path import Path
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

CONTACT_FACE_TO_PSICOFFSET = {
    'front': 0.0,
    'back' : np.pi,
    'left' : 0.5*np.pi,
    'right': -0.5*np.pi
}

# The Slider Convertor
#  -------------------------------------------------------------------
class SliderConvertor(object):
    def __init__(self, opt, curve: sliding_pack.bspline.bspline_curve) -> None:
        self.beta = opt['beta']
        self.dt = opt['dt']
        self.N_min = opt['N_min']
        self.N_MPC = opt['N_MPC']

        self.tracking_config = opt['tracking_config']

        self.path_seg_list = opt['path_seg_list']
        self.num_path_seg = opt['num_path_seg']

        self.curve = curve

        print('There are {0} path segments in total!'.format(self.num_path_seg))

    def get_psic_lim(self):
        """
        Get the psic limit
        """
        return 0.9

    def get_beta(self):
        """
        Get the beta
        """
        return self.beta

    def set_patches(self, ax, x_data, x_p_data, beta, curve_func):
        t_vals = np.linspace(0, 1, 100)
        contour_pts = np.array([curve_func(t) for t in t_vals]).reshape(-1, 2)
        R_pusher = beta[0]
        x0 = x_data[:, 0]
        contour_pts[:, 0] += x0[0]
        contour_pts[:, 1] += x0[1]
        contour_path = Path(contour_pts)
        self.slider = patches.PathPatch(contour_path, lw=2)
        self.pusher = patches.Circle(
                x_p_data[:, 0], radius=R_pusher, facecolor='#7f7f7f', edgecolor='black')
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        self.path_past.set_linewidth(2)

    def animate(self, i, ax, x_data, x_p_data, X_future=None):
        xi = x_data[:, i]
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        transf_i = transforms.Affine2D().translate(xi[0], xi[1]).rotate_around(xi[0], xi[1], xi[2])
        # Set changes
        self.slider.set_transform(transf_i + trans_ax)
        self.pusher.set_center(x_p_data[:, i])
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
        return []

    def track_path_seg(self, i, psic_offset):
        """
        Track the ith path segment
        :param i: path segment index
        :param x_init: the init point
        """
        print('---------------------------------')
        print('Tracking the {0}th path segment!'.format(i+1))
        contact_face = self.path_seg_list[i]['contact_face'][1]
        psic_offset = CONTACT_FACE_TO_PSICOFFSET[contact_face]

        psic_lim = self.get_psic_lim()
        beta = self.get_beta()

        if i == 0:
            X_nom = np.array(self.path_seg_list[i]['X_slider'])  # (N, 4)
            U_nom = np.array(self.path_seg_list[i]['U_slider'])  # (N, 3)
        else:
            X_nom = np.array([self.path_seg_list[i-1]['X_slider'][-1]]+self.path_seg_list[i]['X_slider'])  # (N, 4)
            U_nom = np.array([self.path_seg_list[i-1]['U_slider'][-1]]+self.path_seg_list[i]['U_slider'])  # (N, 3)
        # X_nom[:, 2] = sliding_pack.utils.make_angle_continuous(X_nom[:, 2])
        # X_nom[:, 3] = sliding_pack.utils.make_angle_continuous(X_nom[:, 3])

        # # make sure x0 and x_nom are adjacent on the number line
        # # X_nom[:, 2] += sliding_pack.utils.angle_diff(X_nom[0, 2], x_init[2])
        # X_nom[:, 2] += x_init[2] - X_nom[0, 2]
        
        ## Upsample
        N_nom_pts = len(X_nom)
        if N_nom_pts < self.N_min:
            X_nom = sliding_pack.utils.interpolate_path(X_nom, self.N_min)
        Nidx = len(X_nom)
        
        ## Time Varying Time Horizon
        # if Nidx >= (self.N_MPC/0.6):
        #     _N_MPC = self.N_MPC
        # else:
        #     # _N_MPC = round(0.6*Nidx)
        #     _N_MPC = 3
        _N_MPC = self.N_MPC
        print('N_MPC: ', _N_MPC)
        
        # X_nom_mpc = X_nom[-1, :] + np.repeat((X_nom[-1, :]-X_nom[-2, :]).reshape(1, -1), _N_MPC, axis=0).cumsum(axis=0)
        X_nom_mpc = np.repeat(X_nom[-1, :].reshape(1, -1), _N_MPC, axis=1).reshape(-1, _N_MPC).T
        X_nom = np.concatenate((X_nom, X_nom_mpc), axis=0)
        X_nom[:, 3] = 0.
        traj_pts = X_nom[:, :2]
        GOAL = X_nom[-1, :3]
        X_nom_DM = cs.DM(X_nom.T)

        dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
            configDict=self.tracking_config['dynamics'],
            curve=self.curve,
            contactMode=self.tracking_config['TO']['contactMode'],
            pusherAngleLim=psic_lim
        )

        optObj = sliding_pack.to.buildOptObj(
            dyn_class=dyn,
            timeHorizon=_N_MPC,
            configDict=self.tracking_config['TO'],
            X_nom_val=X_nom_DM,
            U_nom_val=None,
            dt=self.dt,
            max_iter=80
        )

        X_opt = np.empty([dyn.Nx, Nidx-1])
        X_future = np.empty([dyn.Nx, _N_MPC, Nidx-1])
        U_opt = np.empty([dyn.Nu, Nidx-1])
        cost_plot = np.empty([Nidx-1, 1])

        x_init_val = X_nom[0, :]
        X_goal = X_nom[-1, :3]

        x0 = x_init_val.tolist()
        X_opt[:, 0] = x0
        u_opt_last = None

        for idx in range(Nidx-1):
            print("======")
            print(f"{idx=}")
            U_warmStart = None
            if u_opt_last is not None:
                U_warmStart = cs.horzcat(u_opt_last[:, 1:], cs.GenDM_zeros(dyn.Nu, 1))
            resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
                    idx, x0, beta,
                    U_warmStart=U_warmStart, S_goal_val=None,
                    obsCentre=None, obsRadius=None)
            u0 = u_opt[:, 0].elements()
            u_opt_last = u_opt

            x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()
            x0[-1] -= psic_offset
            X_opt[:, idx] = x0
            U_opt[:, idx] = u0
            cost_plot[idx] = f_opt
            X_future[:, :, idx] = np.array(x_opt)
            print('u0: ', [u0[0], u0[1], u0[2]-u0[3]])
            print('f_opt: ', f_opt)

        X_opt = np.array(X_opt).T
        X_future = np.array(X_future).transpose(1, 2, 0)
        U_opt = np.array(U_opt).T
        cost_plot = np.array(cost_plot)

        p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
        p_map = p_new.map(N)
        X_pusher_opt = p_map(np.concatenate((x_init_val.reshape(-1,1), X_opt), axis=1)).toarray()[:, 1:]

        ## Downsample
        uni_idx = np.round(np.linspace(0, Nidx-2, N_nom_pts-1)).astype('int')
        X_opt = X_opt[:, uni_idx]
        U_opt = U_opt[:, uni_idx]
        X_pusher_opt = X_pusher_opt[:, uni_idx]
        X_future = X_future[:, :, uni_idx]
        cost_plot = cost_plot[uni_idx]

        if i == 0:
            X_opt = np.concatenate((x_init_val.reshape(-1,1), X_opt), axis=-1)
            U_opt = np.concatenate((np.zeros(dyn.Nu).reshape(-1,1), U_opt), axis=-1)
            xp0 = p_new(x_init_val)
            X_pusher_opt = np.concatenate((np.array(xp0).reshape(-1,1), X_pusher_opt), axis=-1)
            x_ahead0 = np.repeat(x_init_val.reshape(-1,1), _N_MPC, axis=1)
            X_future = np.concatenate((np.expand_dims(x_ahead0, axis=-1), X_future), axis=-1)
            cost_plot = np.concatenate(([0], cost_plot), axis=0)

        return X_opt, X_nom, U_opt, X_pusher_opt, X_future, cost_plot

    def solve(self):
        """
        Solve the hybrid control problem
        """
        for i in range(self.num_path_seg):
            contact_face = self.path_seg_list[i]['contact_face'][0]
            psic_offset = CONTACT_FACE_TO_PSICOFFSET[contact_face]
            X_opt, X_nom, U_opt, X_pusher_opt, X_future, cost_plot = self.track_path_seg(i, psic_offset)

            if i == 0:
                X_slider = X_opt.copy()
                X_nominal = X_nom.copy()
                U_slider = U_opt.copy()
                X_pusher = X_pusher_opt.copy()
                X_ahead = X_future.copy()
                C_opt = cost_plot.copy()
            else:
                X_slider = np.concatenate((X_slider, X_opt), axis=-1)
                X_nominal = np.concatenate((X_nominal, X_nom), axis=-1)
                U_slider = np.concatenate((U_slider, U_opt), axis=-1)
                X_pusher = np.concatenate((X_pusher, X_pusher_opt), axis=-1)
                X_ahead = np.concatenate((X_ahead, X_future), axis=-1)
                C_opt = np.concatenate((C_opt, cost_plot), axis=0)

        return X_slider, X_nominal, U_slider, X_pusher, X_ahead, C_opt

# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('hardware_frenchs_tracking.yaml')
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 10  # time of the simulation is seconds
freq = 25  # number of increments per second
# N_MPC = 12 # time horizon for the MPC controller
N_MPC = 25  # time horizon for the MPC controller
show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
Nidx = int(N)
idxDist = 5.*freq
# Nidx = 10
#  -------------------------------------------------------------------

import pickle
timestamp = '2025_03_09_15_48'
path_seg = pickle.load(open(os.path.join(r3t_root_dir, 'data', 'wshf', timestamp, 'path_seg.pkl'), 'rb'))
slider_geom = [
    tracking_config['dynamics']['pusherRadious']
]

control_points = tracking_config['dynamics']['control_points']
control_points = np.array(control_points)
curve = sliding_pack.bspline.bspline_curve(control_points)

solver_opt = {
    'beta': slider_geom,
    'dt': dt,
    'N_min': N_MPC,
    'N_MPC': N_MPC,
    'tracking_config': tracking_config,
    'path_seg_list': path_seg,
    'num_path_seg': len(path_seg)
}

solver = SliderConvertor(opt=solver_opt, curve=curve)
X_slider, X_nominal, U_slider, X_pusher, X_ahead, C_opt = solver.solve()

# plot figures and animation
fig, ax = sliding_pack.plots.plot_nominal_traj(
    X_nominal[0,:], X_nominal[1,:], plot_title='pushing tracking'
)
fig.set_size_inches(8, 6, forward=True)
solver.set_patches(ax, X_slider, X_pusher, slider_geom, curve.curve_func)
anim = animation.FuncAnimation(
    fig,
    solver.animate,
    fargs=(ax, X_slider, X_pusher, X_ahead),
    frames=X_slider.shape[1], 
    interval=dt*1000,
    blit=True,
    repeat=False
)
anim.save('./video/R3T_contact_track_' + time.strftime("%Y%m%d-%H%M%S") + '.mp4', fps=25, extra_args=['-vcodec', 'mpeg4'])

plt.show()

exit(0)

# if save_to_file:
#     #  Save data to file using pandas
#     #  -------------------------------------------------------------------
#     df_state = pd.DataFrame(
#                     np.concatenate((
#                         np.array(X_nom_val[:, :Nidx]).transpose(),
#                         np.array(X_plot).transpose(),
#                         np.array(X_pusher_opt).transpose()
#                         ), axis=1),
#                     columns=['x_nom', 'y_nom', 'theta_nom', 'psi_nom',
#                              'x_opt', 'y_opt', 'theta_opt', 'psi_opt',
#                              'x_pusher', 'y_pusher'])
#     df_state.index.name = 'idx'
#     df_state.to_csv('tracking_circle_cc_state.csv',
#                     float_format='%.5f')
#     time = np.linspace(0., T, Nidx-1)
#     print('********************')
#     print(U_plot.transpose().shape)
#     print(cost_plot.shape)
#     print(comp_time.shape)
#     print(time.shape)
#     print(time[:, None].shape)
#     df_action = pd.DataFrame(
#                     np.concatenate((
#                         U_plot.transpose(),
#                         time[:, None],
#                         cost_plot,
#                         comp_time
#                         ), axis=1),
#                     columns=['u0', 'u1', 'u3', 'u4',
#                     # columns=['u0', 'u1', 'u3',
#                              'time', 'cost', 'comp_time'])
#     df_action.index.name = 'idx'
#     df_action.to_csv('tracking_circle_cc_action.csv',
#                      float_format='%.5f')
#     #  -------------------------------------------------------------------

# # Plot Optimization Results
# #  -------------------------------------------------------------------
# fig, axs = plt.subplots(3, 4, sharex=True)
# fig.set_size_inches(10, 10, forward=True)
# t_Nx = np.linspace(0, T, N)
# t_Nu = np.linspace(0, T, N-1)
# t_idx_x = t_Nx[0:Nidx]
# t_idx_u = t_Nx[0:Nidx-1]
# ctrl_g_idx = dyn.g_u.map(Nidx-1)
# ctrl_g_val = ctrl_g_idx(U_plot, del_plot)
# #  -------------------------------------------------------------------
# # plot position
# for i in range(dyn.Nx):
#     axs[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
#                    linestyle='--', label='nom')
#     axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
#                    linestyle='--', label='plan')
#     axs[0, i].plot(t_idx_x, X_plot[i, :], color='orange', label='mpc')
#     handles, labels = axs[0, i].get_legend_handles_labels()
#     axs[0, i].legend(handles, labels)
#     axs[0, i].set_xlabel('time [s]')
#     axs[0, i].set_ylabel('x%d' % i)
#     axs[0, i].grid()
# #  -------------------------------------------------------------------
# # plot computation time
# axs[1, 0].plot(t_idx_u, comp_time, color='b')
# handles, labels = axs[1, 0].get_legend_handles_labels()
# axs[1, 0].legend(handles, labels)
# axs[1, 0].set_xlabel('time [s]')
# axs[1, 0].set_ylabel('comp time [s]')
# axs[1, 0].grid()
# #  -------------------------------------------------------------------
# # plot computation cost
# axs[1, 1].plot(t_idx_u, cost_plot, color='b', label='cost')
# handles, labels = axs[1, 1].get_legend_handles_labels()
# axs[1, 1].legend(handles, labels)
# axs[1, 1].set_xlabel('time [s]')
# axs[1, 1].set_ylabel('cost')
# axs[1, 1].grid()
# #  -------------------------------------------------------------------
# # plot extra variables
# for i in range(dyn.Nz):
#     axs[1, 2].plot(t_idx_u, del_plot[i, :].T, label='s%d' % i)
# handles, labels = axs[1, 2].get_legend_handles_labels()
# axs[1, 2].legend(handles, labels)
# axs[1, 2].set_xlabel('time [s]')
# axs[1, 2].set_ylabel('extra vars')
# axs[1, 2].grid()
# #  -------------------------------------------------------------------
# # plot constraints
# for i in range(dyn.Ng_u):
#     axs[1, 3].plot(t_idx_u, ctrl_g_val[i, :].T, label='g%d' % i)
# handles, labels = axs[1, 3].get_legend_handles_labels()
# axs[1, 3].legend(handles, labels)
# axs[1, 3].set_xlabel('time [s]')
# axs[1, 3].set_ylabel('constraints')
# axs[1, 3].grid()
# #  -------------------------------------------------------------------
# # plot actions
# for i in range(dyn.Nu):
#     axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
#                    linestyle='--', label='plan')
#     axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
#     handles, labels = axs[2, i].get_legend_handles_labels()
#     axs[2, i].legend(handles, labels)
#     axs[2, i].set_xlabel('time [s]')
#     axs[2, i].set_ylabel('u%d' % i)
#     axs[2, i].grid()
# #  -------------------------------------------------------------------

# #  -------------------------------------------------------------------
# plt.show()
# #  -------------------------------------------------------------------

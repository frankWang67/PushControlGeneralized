import warnings
warnings.filterwarnings('ignore')

#  import libraries
#  -------------------------------------------------------------------
import sys
import yaml
import math
import numpy as np
import pandas as pd
import time
from datetime import datetime
import casadi as cs
from matplotlib import patches, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from utils.ur5_interface import *
#  -------------------------------------------------------------------

CONTACT_FACE_TO_AXIS = {
                        'front': '+x',
                        'back': '-x',
                        'left': '+y',
                        'right': '-y'
                       }

# The Slider Convertor
#  -------------------------------------------------------------------
class SliderConvertor(object):
    def __init__(self, opt) -> None:
        # initialize path segment
        self.beta = opt['beta']
        self.dt = opt['dt']
        self.freq = opt['freq']
        self.N_MPC = opt['N_MPC']
        self.v_d = opt['v_d']

        self.tracking_config = opt['tracking_config']

        self.path_seg_list = opt['path_seg_list']
        self.num_path_seg = opt['num_path_seg']

        print('There are {0} path segments in total!'.format(self.num_path_seg))

        # initialize ur5 control interface
        self.ur5_control = initialize_tf_and_control(self.beta, self.dt)

    def get_psic_lim(self, contact_face):
        """
        Get the psic limit on contact face
        :param contact_face: the contact face
        """
        xl, yl = self.beta[:2]
        if contact_face in ['+x', '-x']:
            psic_lim = np.arctan2(0.5*yl, 0.5*xl)
        elif contact_face in ['+y', '-y']:
            psic_lim = np.arctan2(0.5*xl-0.005, 0.5*yl)
        else:
            raise NotImplementedError('SliderConvertor: contact face {0} not supported!'.format(contact_face))
        return psic_lim

    def get_beta(self, contact_face):
        """
        Get the psic limit on contact face
        :param contact_face: the contact face
        """
        xl, yl, rl = self.beta
        if contact_face in ['+x', '-x']:
            beta = [xl, yl, rl]
        elif contact_face in ['+y', '-y']:
            beta = [yl, xl, rl]
        else:
            raise NotImplementedError('SliderConvertor: contact face {0} not supported!'.format(contact_face))
        return beta

    def get_theta_offset_in_path_frame(self, contact_face):
        """
        The X_opt includes theta in the slider's local frame,
            this function computes the theta offset to convert
            it into the path's gloabal frame.
        :param contact_face: the contact face
        """
        if contact_face == '+x':
            bias = np.pi
        elif contact_face == '-x':
            bias = 0.
        elif contact_face == '+y':
            bias = -0.5*np.pi
        elif contact_face == '-y':
            bias = 0.5*np.pi
        else:
            raise NotImplementedError('SliderConvertor: contact face {0} not supported!'.format(contact_face))
        return bias

    def compute_path_seg_point_num(self, X_nom):
        N_pts = len(X_nom)
        path_seg_length = 0.
        for idx_pt in range(N_pts-1):
            path_seg_length += np.linalg.norm(X_nom[idx_pt+1,0:2]-X_nom[idx_pt,0:2], ord=2)
        execution_time = path_seg_length / self.v_d
        path_seg_point_num = math.ceil(self.freq * execution_time)
        path_seg_point_num = int(path_seg_point_num)
        return path_seg_point_num

    def set_patches(self, ax, x_data, x_p_data, beta):
        Xl = beta[0]
        Yl = beta[1]
        R_pusher = beta[2]
        x0 = x_data[:, 0]
        R0 = np.eye(3)
        d0 = R0.dot(np.array([-Xl/2., -Yl/2., 0]))
        self.slider = patches.Rectangle(
                x0[0:2]+d0[0:2], Xl, Yl, angle=0.0, facecolor='#1f77b4', edgecolor='black')
        self.pusher = patches.Circle(
                x_p_data[:, 0], radius=R_pusher, facecolor='#7f7f7f', edgecolor='black')
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        self.path_past.set_linewidth(2)

    def animate(self, i, ax, x_data, x_p_data, beta, X_future=None):
        Xl = beta[0]
        Yl = beta[1]
        xi = x_data[:, i]
        # distance between centre of square reference corner
        Ri = np.block([[sliding_pack.utils.rotation_matrix(xi[2]), np.zeros((2,1))], [np.zeros(2,), 1.]])
        di = Ri.dot(np.array([-Xl/2, -Yl/2, 0]))
        # square reference corner
        ci = xi[0:3] + di
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(
                coords[0], coords[1], xi[2])
        # Set changes
        self.slider.set_transform(trans_ax+trans_i)
        self.slider.set_xy([ci[0], ci[1]])
        self.pusher.set_center(x_p_data[:, i])
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
        return []

    def track_path_seg(self, i, x_init):
        """
        Track the ith path segment
        :param i: path segment index
        :param x_init: the init point
        """
        print('Tracking the {0}th path segment!'.format(i+1))
        contact_face_script = self.path_seg_list[i]['contact_face'][0]  # 'front', 'back', 'left', 'right'
        contact_face = CONTACT_FACE_TO_AXIS[contact_face_script]  # '+x', '-x', '+y', '-y'

        # import pdb; pdb.set_trace()  # tag: for debug only

        # tag: panda control
        if i == 0:
            self.ur5_control.ur5_move_to_contact_face(contact_face_script)
        else:
            self.ur5_control.ur5_move_to_contact_face(contact_face_script, god_view_just_lift_up=True)

        # get x_init from perception
        x_init = self.ur5_control.ur5_get_state_variable(contact_face_script)

        # make for the difference between local frame and global frame
        theta_offset = self.get_theta_offset_in_path_frame(contact_face)

        psic_lim = self.get_psic_lim(contact_face)
        beta = self.get_beta(contact_face)

        if i == 0:
            X_nom = np.array(self.path_seg_list[i]['X_slider'])  # (N, 4)
        else:
            X_nom = np.array([self.path_seg_list[i-1]['X_slider'][-1]]+self.path_seg_list[i]['X_slider'])  # (N, 4)
        X_nom[:, 2] = sliding_pack.utils.make_angle_continuous(X_nom[:, 2])
        X_nom[:, 3] = sliding_pack.utils.make_angle_continuous(X_nom[:, 3])

        # make sure x0 and x_nom are adjacent on the number axis
        # X_nom[:, 2] += sliding_pack.utils.angle_diff(X_nom[0, 2], x_init[2])
        theta_nom_offset = (x_init[2] + sliding_pack.utils.angle_diff(x_init[2], X_nom[0, 2])) - X_nom[0, 2]
        X_nom[:, 2] += theta_nom_offset
        X_nom[:, 2] += theta_offset
        x_init[2] += theta_offset
        
        ## Upsample
        N_nom_pts = len(X_nom)
        N_upsample = self.compute_path_seg_point_num(X_nom)
        X_nom = sliding_pack.utils.interpolate_path(X_nom, N_upsample)
        Nidx = len(X_nom)

        ## Time Varying Time Horizon
        # if Nidx >= (self.N_MPC/0.6):
        #     _N_MPC = self.N_MPC
        # else:
        #     # _N_MPC = round(0.6*Nidx)
        #     _N_MPC = 3
        _N_MPC = self.N_MPC
        print('N_MPC: ', _N_MPC)
        
        X_nom_mpc = X_nom[-1, :] + np.repeat((X_nom[-1, :]-X_nom[-2, :]).reshape(1, -1), _N_MPC, axis=0).cumsum(axis=0)
        X_nom = np.concatenate((X_nom, X_nom_mpc), axis=0)
        X_nom[:, 3] = 0.
        X_nom_DM = cs.DM(X_nom.T)

        dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
            configDict=self.tracking_config['dynamics'],
            contactMode=self.tracking_config['TO']['contactMode'],
            contactFace=contact_face,
            pusherAngleLim=psic_lim
        )

        L_surf = dyn.A(beta).toarray() * L_SURF_SCALE
        self.ur5_control._set_L_surf(L_surf)

        optObj = sliding_pack.to.buildOptObj(
            dyn_class=dyn,
            timeHorizon=_N_MPC,
            configDict=self.tracking_config['TO'],
            X_nom_val=X_nom_DM,
            U_nom_val=None,
            dt=self.dt
        )

        X_opt = np.empty([dyn.Nx, Nidx-1])
        X_future = np.empty([dyn.Nx, _N_MPC, Nidx-1])
        U_opt = np.empty([dyn.Nu, Nidx-1])
        cost_plot = np.empty([Nidx-1, 1])

        x0_last = x_init.copy()
        _rate = rospy.Rate(self.freq)
        # X_opt[:, 0] = x0
        # import pdb; pdb.set_trace()  # tag: for debug only
        for idx in range(Nidx-1):
            loop_start_time = time.time()

            # tag: panda control
            x0 = self.ur5_control.ur5_get_state_variable(contact_face_script)
            x0[2] += theta_offset
            x0[2] = x0_last[2] + sliding_pack.utils.angle_diff(x0_last[2], x0[2])
            x0_last = x0.copy()

            resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
                    idx, x0, beta, d_hat=np.zeros(dyn.Nx).tolist(),
                    S_goal_val=None,
                    obsCentre=None, obsRadius=None)
            u0 = u_opt[:, 0].elements()
            # x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()

            # tag: panda control
            self.ur5_control.ur5_apply_velocity_control(u0, contact_face_script)

            X_opt[:, idx] = x0
            U_opt[:, idx] = u0
            cost_plot[idx] = f_opt
            X_future[:, :, idx] = np.array(x_opt)
            print('u0: ', [u0[0], u0[1], u0[2]-u0[3]])
            print('f_opt: ', f_opt)

            _rate.sleep()
            print('current loop time: {0}(s)'.format(time.time()-loop_start_time))

        # tag: panda control
        self.ur5_control.ur5_stop_move()
        # import pdb; pdb.set_trace()  # tag: for debug only

        p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
        p_map = p_new.map(Nidx)
        X_pusher_opt = p_map(np.concatenate((np.array(x_init).reshape(-1,1), X_opt), axis=1)).toarray()[:, 1:]

        X_opt[2, :] -= theta_offset
        X_future[2, :] -= theta_offset

        ## Downsample
        uni_idx = np.round(np.linspace(0, Nidx-2, N_nom_pts-1)).astype('int')
        X_opt = X_opt[:, uni_idx]
        U_opt = U_opt[:, uni_idx]
        X_pusher_opt = X_pusher_opt[:, uni_idx]
        X_future = X_future[:, :, uni_idx]
        cost_plot = cost_plot[uni_idx, :]

        # add x_init
        if i == 0:
            X_opt = np.concatenate((np.array(x_init).reshape(-1,1), X_opt), axis=-1)
            U_opt = np.concatenate((np.zeros(dyn.Nu).reshape(-1,1), U_opt), axis=-1)
            xp0 = p_new(np.array(x_init)+np.array([0., 0., theta_offset, 0.]))
            X_pusher_opt = np.concatenate((np.array(xp0).reshape(-1,1), X_pusher_opt), axis=-1)
            x_ahead0 = np.repeat(np.array(x_init).reshape(-1,1), _N_MPC, axis=1)
            X_future = np.concatenate((np.expand_dims(x_ahead0, axis=-1), X_future), axis=-1)
            cost_plot = np.concatenate(([[0]], cost_plot), axis=0)

            uni_idx_nom = np.round(np.linspace(0, Nidx-1, N_nom_pts)).astype('int')
            X_nom = X_nom[:-_N_MPC, :][uni_idx_nom, :].T
        else:
            X_nom = X_nom[:-_N_MPC, :][uni_idx, :].T

        return X_opt, X_nom, U_opt, X_pusher_opt, X_future, cost_plot

    def solve(self, x_init):
        """
        Solve the hybrid control problem
        """
        for i in range(self.num_path_seg):
            X_opt, X_nom, U_opt, X_pusher_opt, X_future, cost_plot = self.track_path_seg(i, x_init)
            contact_face = self.path_seg_list[i]['contact_face'][0]
            contact_face = CONTACT_FACE_TO_AXIS[contact_face]
            theta_offset = self.get_theta_offset_in_path_frame(contact_face)

            # import pdb; pdb.set_trace()  # tag: for debug only
            # X_opt[2, :] += theta_offset

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

            x_init = X_opt[:, -1].tolist()
            x_init[-1] = 0.

        return X_slider, X_nominal, U_slider, X_pusher, X_ahead, C_opt

# tag: the main function
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('tracking_robot.yaml')
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 10  # time of the simulation is seconds
freq = 25  # number of increments per second
N_MPC = 30  # time horizon for the MPC controller
show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
#  -------------------------------------------------------------------
# Select Planning Scene
#  -------------------------------------------------------------------
UR5_PLANNING_SCENE = 4
#  -------------------------------------------------------------------

import pickle
path_seg = pickle.load(open('/home/roboticslab/jyp/pusher_slider/data/traj/saved/path_seg_ur5_#{0}.pkl'.format(UR5_PLANNING_SCENE), 'rb'))
slider_geom = [
                tracking_config['dynamics']['xLenght'],
                tracking_config['dynamics']['yLenght'],
                tracking_config['dynamics']['pusherRadious']
              ]

solver_opt = {
    'beta': slider_geom,
    'dt': dt,
    'freq': freq,
    'N_MPC': N_MPC,
    'v_d': 0.01,  # desired tracking velocity [m/s]
    'tracking_config': tracking_config,
    'path_seg_list': path_seg,
    'num_path_seg': len(path_seg)
}

import pdb; pdb.set_trace()

solver = SliderConvertor(opt=solver_opt)
X_slider, X_nominal, U_slider, X_pusher, X_ahead, C_opt = \
    solver.solve(x_init=None)

import pdb; pdb.set_trace()

# plot figures and animation
fig, ax = sliding_pack.plots.plot_nominal_traj(
    X_nominal[0,:], X_nominal[1,:], plot_title='pushing tracking'
)
fig.set_size_inches(8, 6, forward=True)
solver.set_patches(ax, X_slider, X_pusher, beta=slider_geom)
anim = animation.FuncAnimation(
    fig,
    solver.animate,
    fargs=(ax, X_slider, X_pusher, slider_geom, X_ahead),
    frames=X_slider.shape[1], 
    interval=dt*1000,
    blit=True,
    repeat=False
)
anim.save('/home/roboticslab/jyp/pusher_slider/data/video/iros_tracking_push_away1.mp4', fps=25, extra_args=['-vcodec', 'libx264'])

plt.show()

# save data
try:
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    data_save_path = '/home/roboticslab/jyp/pusher_slider/data/result/' + 'track_ur5_#{0}_'.format(UR5_PLANNING_SCENE) + timestamp
    data_to_be_saved = {'X_slider': X_slider,
                        'X_nominal': X_nominal,
                        'U_slider': U_slider,
                        'X_pusher': X_pusher,
                        'X_ahead': X_ahead,
                        'C_opt': C_opt}
    pickle.dump(data_to_be_saved, open(data_save_path, 'wb'))
except:
    import pdb; pdb.set_trace()  # tag: in case of data save error

exit(0)

if save_to_file:
    #  Save data to file using pandas
    #  -------------------------------------------------------------------
    df_state = pd.DataFrame(
                    np.concatenate((
                        np.array(X_nom_val[:, :Nidx]).transpose(),
                        np.array(X_plot).transpose(),
                        np.array(X_pusher_opt).transpose()
                        ), axis=1),
                    columns=['x_nom', 'y_nom', 'theta_nom', 'psi_nom',
                             'x_opt', 'y_opt', 'theta_opt', 'psi_opt',
                             'x_pusher', 'y_pusher'])
    df_state.index.name = 'idx'
    df_state.to_csv('tracking_circle_cc_state.csv',
                    float_format='%.5f')
    time = np.linspace(0., T, Nidx-1)
    print('********************')
    print(U_plot.transpose().shape)
    print(cost_plot.shape)
    print(comp_time.shape)
    print(time.shape)
    print(time[:, None].shape)
    df_action = pd.DataFrame(
                    np.concatenate((
                        U_plot.transpose(),
                        time[:, None],
                        cost_plot,
                        comp_time
                        ), axis=1),
                    columns=['u0', 'u1', 'u3', 'u4',
                    # columns=['u0', 'u1', 'u3',
                             'time', 'cost', 'comp_time'])
    df_action.index.name = 'idx'
    df_action.to_csv('tracking_circle_cc_action.csv',
                     float_format='%.5f')
    #  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
if show_anim:
    #  ---------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(
                x0_nom[:Nidx], x1_nom[:Nidx], plot_title='')
    # add computed nominal trajectory
    X_nom_val_opt = np.array(X_nom_val_opt)
    # ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
    #         linewidth=2.0, linestyle='dashed')
    X_nom_comp = np.array(X_nom_comp)
    # ax.plot(X_nom_comp[0, :], X_nom_comp[1, :], color='green',
    #         linewidth=2.0, linestyle='dashed')
    # add obstacles
    if optObj.numObs > 0:
        for i in range(len(obsCentre)):
            circle_i = plt.Circle(obsCentre[i], obsRadius[i], color='b')
            ax.add_patch(circle_i)
    # set window size
    fig.set_size_inches(8, 6, forward=True)
    # get slider and pusher patches
    dyn.set_patches(ax, X_plot, beta)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            dyn.animate,
            fargs=(ax, X_plot, beta, X_future),
            frames=Nidx-1,
            interval=dt*1000,  # microseconds
            blit=True,
            repeat=False,
    )
    # to save animation, uncomment the line below:
    # ani.save('MPC_MPCC_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 4, sharex=True)
fig.set_size_inches(10, 10, forward=True)
t_Nx = np.linspace(0, T, N)
t_Nu = np.linspace(0, T, N-1)
t_idx_x = t_Nx[0:Nidx]
t_idx_u = t_Nx[0:Nidx-1]
ctrl_g_idx = dyn.g_u.map(Nidx-1)
ctrl_g_val = ctrl_g_idx(U_plot, del_plot)
#  -------------------------------------------------------------------
# plot position
for i in range(dyn.Nx):
    axs[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
                   linestyle='--', label='nom')
    axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
                   linestyle='--', label='plan')
    axs[0, i].plot(t_idx_x, X_plot[i, :], color='orange', label='mpc')
    handles, labels = axs[0, i].get_legend_handles_labels()
    axs[0, i].legend(handles, labels)
    axs[0, i].set_xlabel('time [s]')
    axs[0, i].set_ylabel('x%d' % i)
    axs[0, i].grid()
#  -------------------------------------------------------------------
# plot computation time
axs[1, 0].plot(t_idx_u, comp_time, color='b')
handles, labels = axs[1, 0].get_legend_handles_labels()
axs[1, 0].legend(handles, labels)
axs[1, 0].set_xlabel('time [s]')
axs[1, 0].set_ylabel('comp time [s]')
axs[1, 0].grid()
#  -------------------------------------------------------------------
# plot computation cost
axs[1, 1].plot(t_idx_u, cost_plot, color='b', label='cost')
handles, labels = axs[1, 1].get_legend_handles_labels()
axs[1, 1].legend(handles, labels)
axs[1, 1].set_xlabel('time [s]')
axs[1, 1].set_ylabel('cost')
axs[1, 1].grid()
#  -------------------------------------------------------------------
# plot extra variables
for i in range(dyn.Nz):
    axs[1, 2].plot(t_idx_u, del_plot[i, :].T, label='s%d' % i)
handles, labels = axs[1, 2].get_legend_handles_labels()
axs[1, 2].legend(handles, labels)
axs[1, 2].set_xlabel('time [s]')
axs[1, 2].set_ylabel('extra vars')
axs[1, 2].grid()
#  -------------------------------------------------------------------
# plot constraints
for i in range(dyn.Ng_u):
    axs[1, 3].plot(t_idx_u, ctrl_g_val[i, :].T, label='g%d' % i)
handles, labels = axs[1, 3].get_legend_handles_labels()
axs[1, 3].legend(handles, labels)
axs[1, 3].set_xlabel('time [s]')
axs[1, 3].set_ylabel('constraints')
axs[1, 3].grid()
#  -------------------------------------------------------------------
# plot actions
for i in range(dyn.Nu):
    axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
                   linestyle='--', label='plan')
    axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
    handles, labels = axs[2, i].get_legend_handles_labels()
    axs[2, i].legend(handles, labels)
    axs[2, i].set_xlabel('time [s]')
    axs[2, i].set_ylabel('u%d' % i)
    axs[2, i].grid()
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------

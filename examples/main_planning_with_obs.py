# Author: Joao Moura (Modified by Yongpeng Jiang)
# Date: 11/25/2022
#  -------------------------------------------------------------------
# Description:
#  This script implements a Differential Dynamic Programming with
#  exhaustive tree-search over mode sequences, which is used in
#  plannar pushing task with obstacle avoidance.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
from copy import deepcopy
import sys
import yaml
import numpy as np
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.transforms as transforms
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from rrt_pack.utilities.data_proc import KinoPlanData
from rrt_pack.utilities.planar_plotting import plot_obstacles
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
planning_config = sliding_pack.load_config('planning_switch_config.yaml')
planning_config['TO']['numObs'] = 0
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 2.5  # time of the simulation is seconds
freq = 50  # number of increments per second @TODO
show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
#  -------------------------------------------------------------------


class SliderConvertor(object):
    def __init__(self, configDict) -> None:
        self.freq = configDict['freq']
        self.base_T = configDict['T']
        self.base_path_len = configDict['path_len']
        self.beta = configDict['beta']
        self.planning_config = None

        self.yaw_shift = {'-x': 0., '+x': np.pi, '-y': 0.5*np.pi, '+y': -0.5*np.pi}

        self.path = None  # via point of COG
        self.dubins = None  # dubins center, radius of sub path (theta0, dtheta, center_x, center_y, radious)
        self.path_len = None  # length of sub path
        self.push_pt = None  # contact point on the periphery

        self.X_nom = np.zeros((0, 4))
        self.X_opt = np.zeros((0, 4))
        self.X_con = np.zeros((0, 2))
        self.Face = []
        
    def plan_data_parser(self, data):
        """
        Parse the planning data
        - path: COG trajectory
        - dubins: dubins curve parameters
        - path_len: length of the subpath
        - contact: contact point on the peripheral
        """
        self.path = data['path']
        self.dubins = data['dubins']
        self.path_len = data['path_len']
        self.push_pt = data['contact']

    def planning_config_setter(self, planning_config):
        """
        Set the planning configurations
        """
        self.planning_config = planning_config
        self.numObs = planning_config['TO']['numObs']
        self.obsCentre = None
        self.obsRadius = None

    def timing_decider(self, i):
        """
        Decide time steps and interval for the subpath, according to its length
        """
        dt = 1.0/self.freq
        T = self.base_T * (self.path_len[i] / self.base_path_len)
        N = max(10, int(T*self.freq))

        return dt, N

    def face_decider(self, i):
        """
        Decide which face the pusher is in contact with
        Recalculate psi
        """
        default_beta = self.beta['x_face']
        x, y = self.push_pt[i, 0], self.push_pt[i, 1]
        if np.abs(x - 0.5 * default_beta[0]) < 1e-5:
            psi = -np.arctan2(-y, default_beta[0]/2)
            return '+x', psi
        elif np.abs(x + 0.5 * default_beta[0]) < 1e-5:
            psi = -np.arctan2(y, default_beta[0]/2)
            return '-x', psi
        elif np.abs(y - 0.5 * default_beta[1]) < 1e-5:
            psi = -np.arctan2(x, default_beta[1]/2)
            return '+y', psi
        elif np.abs(y + 0.5 * default_beta[1]) < 1e-5:
            psi = -np.arctan2(-x, default_beta[1]/2)
            return '-y', psi

    def solve(self):
        # x_init: initial state of optimization problem
        # x_init_nom: the first state of nominal states
        x_init = self.path[0, :].tolist()
        x_init_nom = deepcopy(x_init)

        for i in range(len(self.path) - 1):
            dt, N = self.timing_decider(i)  # decide the time step and number of intervals
            face, psi = self.face_decider(i)  # decide the contact face
            dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
                    configDict=self.planning_config['dynamics'],
                    contactMode=self.planning_config['TO']['contactMode'],
                    contactFace=face,
                    pusherAngleLim=psi
                )
            
            # initialize the symbol function p(pusher position)
            if i == 0:
                self.p = dyn.p
                
            # initialize the symbol function R(rotation matrix)
            if i == 0:
                self.R = dyn.R

            yaw_shift = self.yaw_shift[face]
            beta = self.beta['x_face'] if face[-1] == 'x' else self.beta['y_face']

            x_init[2] = x_init[2] + yaw_shift
            if len(x_init) == 3:
                x_init = np.append(x_init, psi)
            else:
                x_init[-1] = psi
                
            x_init_nom[2] = x_init_nom[2] + yaw_shift
            if len(x_init_nom) == 3:
                x_init_nom = np.append(x_init_nom, psi)
            else:
                x_init_nom[-1] = psi
            
            # import pdb; pdb.set_trace()
            
            X_goal = self.path[i + 1, :].tolist()
            X_goal[2] = X_goal[2] + yaw_shift
            X_goal = np.append(X_goal, psi)
            x0_nom, x1_nom = sliding_pack.traj.generate_dubins_curve(self.dubins[i, 0], self.dubins[i, 1], self.dubins[i, 2], self.dubins[i, 3], \
                                                                     self.dubins[i, 4], N, 0)
            X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
            X_nom_val[:2, :] += x_init_nom[:2]
            X_nom_val[2, :] = np.linspace(x_init_nom[2], x_init_nom[2] + self.dubins[i, 1], N)  # the yaw angle was recomputed by linear interpolation
            X_nom_val[3, :] = psi  # the finger contact position was designated by planning data
            
            # if the yaw in nominal state and (init/goal) state are two far
            # bring them together (continuous)
            if np.abs(X_nom_val[2, 0] - x_init[2]) > np.pi:
                x_init[2] = x_init[2] + 2 * np.sign(X_nom_val[2, 0] - x_init[2]) * np.pi
            if np.abs(X_nom_val[2, -1] - X_goal[2]) > np.pi:
                X_goal[2] = X_goal[2] + 2 * np.sign(X_nom_val[2, -1] - X_goal[2]) * np.pi
            
            if type(x_init) == np.ndarray:
                x_init = x_init.tolist()
                
            if type(X_goal) == np.ndarray:
                X_goal = X_goal.tolist()
            
            # X_nom_val[:, 2] += yaw_shift
            optObj = sliding_pack.to.buildOptObj(dyn, N, self.planning_config['TO'], dt=dt, useGoalFlag=True)

            resultFlag, X_nom_val_opt, U_nom_val_opt, other_opt, _, t_opt = optObj.solveProblem(
                    0, x_init, beta,
                    X_warmStart=X_nom_val,
                    obsCentre=self.obsCentre, obsRadius=self.obsRadius,
                    X_goal_val=X_goal)
            
            print('Solved the %dth sub path optimization problem.' % i)

            # compute and store pusher positions
            theta = X_nom_val_opt[2, :].toarray().squeeze()  # yaw angle
            psi = X_nom_val_opt[3, :].toarray().squeeze()  # contact psi angle
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]]).transpose(2, 0, 1)
            pusher_coords = np.array([(-beta[0]/2-beta[2])*np.ones(X_nom_val_opt.shape[1]), -(beta[0]/2)*np.tan(psi)]).transpose()
            if len(pusher_coords.shape) == 2:
                pusher_coords = np.expand_dims(pusher_coords, axis=2)
            pusher_coords = np.matmul(rot_mat, pusher_coords).squeeze() + X_nom_val_opt[:2, :].T
            
            # plt.figure()
            # plt.scatter(X_nom_val_opt.toarray()[0, :], X_nom_val_opt.toarray()[1, :])
            # plt.scatter(X_nom_val.toarray()[0, :], X_nom_val.toarray()[1, :])
            # plt.gca().set_aspect("equal")
            # plt.xlim([0.0, 0.5])
            # plt.ylim([0.0, 0.5])
            # plt.show()
            # import pdb; pdb.set_trace()
            
            X_nom_val_opt[2, :] = X_nom_val_opt[2, :] - yaw_shift
            X_nom_val[2, :] = X_nom_val[2, :] - yaw_shift
            
            self.X_nom = np.concatenate((self.X_nom, X_nom_val.T if i == 0 else X_nom_val.T[1:, :]), axis=0)
            self.X_opt = np.concatenate((self.X_opt, X_nom_val_opt.T if i == 0 else X_nom_val_opt.T[1:, :]), axis=0)
            self.Face = self.Face + [face] * (N if i == 0 else (N - 1))
            
            self.X_con = np.concatenate((self.X_con, pusher_coords if i == 0 else pusher_coords[1:, :]), axis=0)

            # update x_init for next iteration
            x_init = X_nom_val_opt[:, -1].toarray().squeeze()
            x_init_nom = X_nom_val[:, -1].toarray().squeeze()

        return self.X_nom, self.X_opt

    def set_patches(self, ax, x_data, x_data_nom):
        # size parameters
        default_beta = self.beta['x_face']
        Xl = default_beta[0]
        Yl = default_beta[1]
        R_pusher = default_beta[2]
        
        # set patch for real slider
        x0 = x_data[0, :]
        R0 = np.eye(3)
        d0 = R0.dot(np.array([-Xl/2., -Yl/2., 0]))
        self.slider = patches.Rectangle(
                x0[0:2]+d0[0:2], Xl, Yl, angle=0.0, color='#3682be', edgecolor=None, alpha=0.5)
        self.pusher = patches.Circle(
                np.array(self.p(x0, default_beta)), radius=R_pusher, color='black', alpha=0.5)
        
        # set patch for imaginary slider (nominal slider)
        x0_nom = x_data_nom[0, :]
        R0_nom = np.eye(3)
        d0_nom = R0_nom.dot(np.array([-Xl/2., -Yl/2., 0]))
        self.slider_nom = patches.Rectangle(
                x0_nom[0:2]+d0_nom[0:2], Xl, Yl, angle=0.0, color='#45a776', edgecolor=None, alpha=0.5)
        
        # set slider trajectory
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        
        # add patches
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        ax.add_patch(self.slider_nom)
        self.path_past.set_linewidth(2)

    def animate(self, i, ax, x_data, x_data_nom, X_future=None):
        # size parameters
        default_beta = self.beta['x_face']
        Xl = default_beta[0]
        Yl = default_beta[1]
        
        # update real slider
        xi = x_data[i, :]
        Ri = np.array(self.R(xi))  # distance between centre of square reference corner
        di = Ri.dot(np.array([-Xl/2, -Yl/2, 0]))
        ci = xi[0:3] + di  # square reference corner
        trans_ax = ax.transData  # compute transformation with respect to rotation angle xi[2]
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(
                coords[0], coords[1], xi[2])
        # Set changes
        self.slider.set_transform(trans_ax+trans_i)
        self.slider.set_xy([ci[0], ci[1]])
        self.pusher.set_center(self.X_con[i, :])
        
        # update imaginary slider (nominal slider)
        xi_nom = x_data_nom[i, :]
        Ri_nom = np.array(self.R(xi_nom))  # distance between centre of square reference corner
        di_nom = Ri_nom.dot(np.array([-Xl/2, -Yl/2, 0]))
        ci_nom = xi_nom[0:3] + di_nom  # square reference corner
        trans_ax_nom = ax.transData  # compute transformation with respect to rotation angle xi[2]
        coords_nom = trans_ax_nom.transform(ci_nom[0:2])
        trans_i_nom = transforms.Affine2D().rotate_around(
                coords_nom[0], coords_nom[1], xi_nom[2])
        # Set changes
        self.slider_nom.set_transform(trans_ax_nom+trans_i_nom)
        self.slider_nom.set_xy([ci_nom[0], ci_nom[1]])
        
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0:i, 0], x_data[0:i, 1])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
        return []

"""
# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        configDict=planning_config['dynamics'],
        contactMode=planning_config['TO']['contactMode'],
        contactFace='-y',
        pusherAngleLim=0.
)
#  -------------------------------------------------------------------

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
X_goal = planning_config['TO']['X_goal']
X_goal = [0.4, 0.1, 0., 0.]
# print(X_goal)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.3, 0.4, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.1, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.2, N, 0)
#  -------------------------------------------------------------------
# stack state and derivative of state
# the slider rotate to tangent direction of the nominal trajectory in one step
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
optObj = sliding_pack.to.buildOptObj(
        dyn, N, planning_config['TO'], dt=dt, useGoalFlag=True)
# Set obstacles
#  ------------------------------------------------------------------
if optObj.numObs==0:
    obsCentre = None
    obsRadius = None
elif optObj.numObs==2:
    obsCentre = [[0.2, 0.2], [0.1, 0.5]]
    obsRadius = [0.05, 0.05]
elif optObj.numObs==3:
    obsCentre = [[0.2, 0.2], [0.0, 0.4], [0.3, -0.05]]
    obsRadius = [0.05, 0.05, 0.05]
#  ------------------------------------------------------------------
x_init = [0., 0., 0.5*np.pi, 0.]
# x_init = [0., 0., -20.*(np.pi/180.), -50.*(np.pi/180.)]
# x_init = [0.38, 0.22, -70.*(np.pi/180.), 0.]
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]
# x_init = [0., 0., 340.*(np.pi/180.), 0.]
resultFlag, X_nom_val_opt, U_nom_val_opt, other_opt, _, t_opt = optObj.solveProblem(
        0, x_init, beta,
        X_warmStart=X_nom_val,
        obsCentre=obsCentre, obsRadius=obsRadius,
        X_goal_val=X_goal)

import pdb; pdb.set_trace()

f_d = cs.Function('f_d', [dyn.x, dyn.u], [dyn.x + dyn.f(dyn.x, dyn.u, beta)*dt])
f_rollout = f_d.mapaccum(N-1)
print('comp time: ', t_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)
X_pusher_opt = p_map(X_nom_val_opt)
#  ------------------------------------------------------------------


if save_to_file:
    #  Save data to file using pandas
    #  -------------------------------------------------------------------
    df_state = pd.DataFrame(
                    np.array(cs.vertcat(X_nom_val_opt,X_pusher_opt)).transpose(),
                    columns=['x_slider', 'y_slider', 'theta_slider', 'psi_pusher', 'x_pusher', 'y_pusher'])
    df_state.to_csv('planning_positive_angle_state.csv',
                    float_format='%.5f')
    df_action = pd.DataFrame(
                    np.array(U_nom_val_opt).transpose(),
                    columns=['u0', 'u1', 'u3', 'u3'])
    df_action.to_csv('planning_positive_angle_action.csv',
                     float_format='%.5f')
    #  -------------------------------------------------------------------
"""
plan_data = KinoPlanData(filename="rrt_planar_pushing_test")
packed_data = plan_data.data_packer()

converter_config = {'freq': 50, 'T': 2.5, 'path_len':0.4, 
                    'beta': {'x_face': [planning_config['dynamics']['xLenght'],
                                        planning_config['dynamics']['yLenght'],
                                        planning_config['dynamics']['pusherRadious']],
                             'y_face': [planning_config['dynamics']['yLenght'],
                                        planning_config['dynamics']['xLenght'],
                                        planning_config['dynamics']['pusherRadious']]}}
convertor = SliderConvertor(configDict=converter_config)
convertor.planning_config_setter(planning_config)
convertor.plan_data_parser(packed_data)
x_nom, x_opt = convertor.solve()

import pdb; pdb.set_trace()

# Animation
#  -------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
if show_anim:
    #  ---------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(
                x_nom[:, 0], x_nom[:, 1], plot_title='')
    # plot all the via points
    plt.scatter(convertor.path[:, 0], convertor.path[:, 1], marker='x', color='aquamarine')
    # plot obstacles
    plot_obstacles(ax, packed_data['obstacle'])
    # add computed nominal trajectory
    if type(x_nom) is not np.ndarray:
        X_nom_val = np.array(x_nom)
    else:
        X_nom_val = x_nom
    # add solved optimal nominal trajectory
    if type(x_opt) is not np.ndarray:
        X_nom_val_opt = np.array(x_opt)
    else:
        X_nom_val_opt = x_opt
    ax.plot(X_nom_val_opt[:, 0], X_nom_val_opt[:, 1], color='blue',
            linewidth=2.0, linestyle='dashed')
    # add obstacles
    if convertor.numObs > 0:
        for i in range(len(convertor.obsCentre)):
            circle_i = plt.Circle(convertor.obsCentre[i], convertor.obsRadius[i], color='b')
            ax.add_patch(circle_i)
    # set window size
    fig.set_size_inches(8, 6, forward=True)
    # get slider and pusher patches
    # convertor.set_patches(ax, X_nom_val_opt)
    convertor.set_patches(ax, X_nom_val_opt, X_nom_val)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            convertor.animate,
            # fargs=(ax, X_nom_val_opt),
            fargs=(ax, X_nom_val_opt, X_nom_val),
            frames=len(X_nom_val_opt)-1,
            interval=(1./convertor.freq)*1000*10,  # microseconds
            blit=True,
            repeat=False,
    )
    # to save animation, uncomment the line below:
    # ani.save('planning_with_obstacles1.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
    ani.save('planning_with_obstacles1.mp4', fps=25)
#  -------------------------------------------------------------------

# # Plot Optimization Results
# #  -------------------------------------------------------------------
# fig, axs = plt.subplots(3, 4, sharex=True)
# fig.set_size_inches(10, 10, forward=True)
# t_Nx = np.linspace(0, T, N)
# t_Nu = np.linspace(0, T, N-1)
# ctrl_g = dyn.g_u.map(N-1)
# ctrl_g_val = ctrl_g(U_nom_val_opt, other_opt)
# #  -------------------------------------------------------------------
# # plot position
# for i in range(dyn.Nx):
#     axs[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
#                    linestyle='--', label='nom')
#     axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
#                    linestyle='--', label='plan')
#     handles, labels = axs[0, i].get_legend_handles_labels()
#     axs[0, i].legend(handles, labels)
#     axs[0, i].set_xlabel('time [s]')
#     axs[0, i].set_ylabel('x%d' % i)
#     axs[0, i].grid()
# #  -------------------------------------------------------------------
# # plot extra variables
# for i in range(dyn.Nz):
#     axs[1, 2].plot(t_Nu, other_opt[i, :].T, label='s%d' % i)
# handles, labels = axs[1, 2].get_legend_handles_labels()
# axs[1, 2].legend(handles, labels)
# axs[1, 2].set_xlabel('time [s]')
# axs[1, 2].set_ylabel('extra vars')
# axs[1, 2].grid()
# #  -------------------------------------------------------------------
# # plot constraints
# for i in range(dyn.Ng_u):
#     axs[1, 3].plot(t_Nu, ctrl_g_val[i, :].T, label='g%d' % i)
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
#     handles, labels = axs[2, i].get_legend_handles_labels()
#     axs[2, i].legend(handles, labels)
#     axs[2, i].set_xlabel('time [s]')
#     axs[2, i].set_ylabel('u%d' % i)
#     axs[2, i].grid()
# #  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------

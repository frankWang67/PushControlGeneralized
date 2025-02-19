# Author: Joao Moura
# Date: 21/08/2020
#  -------------------------------------------------------------------
# Description:
#  This script implements a non-linear program (NLP) model predictive controller (MPC)
#  for tracking a trajectory of a square slider object with a single
#  and sliding contact pusher.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import time
import numpy as np
# import pandas as pd
import casadi as cs
import scipy.interpolate as spi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('tracking_config.yaml')
# planning_config = sliding_pack.load_config('planning_switch_config.yaml')
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
# T = 10  # time of the simulation is seconds
T = 30  # time of the simulation is seconds

# freq = 15  # number of increments per second
freq = 20  # number of increments per second

# N_MPC = 12 # time horizon for the MPC controller
N_MPC = 30  # time horizon for the MPC controller
# x_init_val = [-0.03, 0.03, 30*(np.pi/180.), 0]

## x_traj from real robot exp
# x_traj = np.load('./data/x_traj.npy')
# u_traj = np.load('./data/u_traj.npy')


x_init_val = [0.0, 0.0, 0.0, 0.0]
# x_init_val = [0., 0., 0.2*np.pi, 0.]
# x_init_val = [0., 0.0, 0.02*np.pi, 0]
# x_init_val = [0.0, 0.0, 0.0, np.pi]
# x_init_val = [0.4241445, 0.01386, -0.0365, 0.]

# x_init_val = x_traj[:, 0].tolist()

psic_offset = np.pi

show_anim = True
save_to_file = False
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
Nidx = int(N)
# idxDist = 15.*freq
# Nidx = 10
#  -------------------------------------------------------------------

# define slider
#  -------------------------------------------------------------------
control_points = tracking_config['dynamics']['control_points']
control_points = np.array(control_points)
curve = sliding_pack.bspline.bspline_curve(control_points)
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        tracking_config['dynamics'],
        curve,
        tracking_config['TO']['contactMode'],
        pusherAngleLim=tracking_config['dynamics']['xFacePsiLimit'],
        limit_surf_gain=1.
)
#  -------------------------------------------------------------------

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
X_goal = tracking_config['TO']['X_goal']
# print(X_goal)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.35, 0.0, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.3, 0.0, 0.05, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, N_MPC)

x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)

# x0_nom, x1_nom = sliding_pack.traj.generate_traj_ellipse(-np.pi/2, 3*np.pi/2, 0.2, 0.1, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.3, N, N_MPC)

## offset nominal traj
# x0_nom, x1_nom = x0_nom+x_init_val[0], x1_nom+x_init_val[1]

#  -------------------------------------------------------------------
# stack state and derivative of state

X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, x_init_val[2], dt)

#  ------------------------------------------------------------------
# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
# dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
#         planning_config['dynamics'],
#         curve_func_nom,
#         tangent_func_nom,
#         normal_func_nom,
#         planning_config['TO']['contactMode'],
#         pusherAngleLim=tracking_config['dynamics']['xFacePsiLimit']
# )
# print("`Sys_sq_slider_quasi_static_ellip_lim_surf` done")
# optObjNom = sliding_pack.to.buildOptObj(
#         dynNom, N+N_MPC, planning_config['TO'], dt=dt, max_iter=150)
# print("`buildOptObj` done")
# beta = [
#     planning_config['dynamics']['x_len'],
#     planning_config['dynamics']['y_len'],
#     planning_config['dynamics']['pusherRadious']
# ]
beta = [
    tracking_config['dynamics']['x_len'],
    tracking_config['dynamics']['y_len'],
    tracking_config['dynamics']['pusherRadious']
]
# print("`solveProblem` start")
# resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjNom.solveProblem(
#         0, [0., 0., 0.*(np.pi/180.), 0.], beta, [0., 0., 0., 0.],
#         X_warmStart=X_nom_val)
# print("`solveProblem` done")
# if dyn.Nu > dynNom.Nu:
#     U_nom_val_opt = cs.vertcat(
#             U_nom_val_opt,
#             cs.DM.zeros(np.abs(dyn.Nu - dynNom.Nu), N+N_MPC-1))
# elif dynNom.Nu > dyn.Nu:
#     U_nom_val_opt = U_nom_val_opt[:dyn.Nu, :]
# f_d = cs.Function('f_d', [dyn.x, dyn.u], [dyn.x + dyn.f(dyn.x, dyn.u, beta)*dt])
# f_rollout = f_d.mapaccum(N+N_MPC-1)
# X_nom_comp = f_rollout([0., 0., 0., 0.], U_nom_val_opt)
#  ------------------------------------------------------------------

# define optimization problem
#  -------------------------------------------------------------------
optObj = sliding_pack.to.buildOptObj(
        dyn, N_MPC, tracking_config['TO'], psic_offset, 
        X_nom_val, None, dt=dt, max_iter=80
)
#  -------------------------------------------------------------------

# Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([dyn.Nx, Nidx])
U_plot = np.empty([dyn.Nu, Nidx-1])
del_plot = np.empty([dyn.Nz, Nidx-1])
X_plot[:, 0] = x_init_val
X_future = np.empty([dyn.Nx, N_MPC, Nidx])
U_future = np.empty([dyn.Nu, N_MPC-1, Nidx])
comp_time = np.empty((Nidx-1, 1))
success = np.empty(Nidx-1)
cost_plot = np.empty((Nidx-1, 1))
#  -------------------------------------------------------------------

#  Set selection matrix for X_goal
#  -------------------------------------------------------------------
if X_goal is None:
    S_goal_val = None
else:
    S_goal_idx = N_MPC-2
    S_goal_val = [0]*(N_MPC-1)
    S_goal_val[S_goal_idx] = 1
#  -------------------------------------------------------------------

# Set obstacles
#  ------------------------------------------------------------------
if optObj.numObs==0:
    obsCentre = None
    obsRadius = None
elif optObj.numObs==1:
    obsCentre = [[-0.27, 0.1]]
    # obsCentre = [[0., 0.28]]
    # obsCentre = [[0.2, 0.2]]
    obsRadius = [0.05]
#  ------------------------------------------------------------------

# Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val

# import pdb
# pdb.set_trace()
# exit(0)

for idx in range(Nidx-1):
    # if idx >= 100:
    #     break
    print('-------------------------')
    print(idx)
    # if idx == idxDist:
    #     print('i died here')
    #     x0[0] += 0.0
    #     x0[1] += -0.03
    #     x0[2] += 0.*(np.pi/180.)
    # ---- solve problem ----
    # x0 = x_traj[:, idx+1].tolist()
    X_plot[:, idx+1] = x0
    # U_warmStart = cs.GenDM_zeros(dyn.Nu, N_MPC-1)
    # U_warmStart[0, :] = 1.0
    U_warmStart = None
    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
            idx, x0, beta, [0., 0., 0., 0.],
            U_warmStart=U_warmStart, S_goal_val=S_goal_val,
            obsCentre=obsCentre, obsRadius=obsRadius)
    print(f"{f_opt=}")
    # import pdb; pdb.set_trace()
    # ---- update initial state (simulation) ----
    u0 = u_opt[:, 0].elements()
    # x0 = x_opt[:,1].elements()
    x0[-1] += psic_offset
    x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()
    x0[-1] -= psic_offset

    ## add noise to x0
    # x0 = np.array(x0) + np.random.uniform(low=[])

    # ---- store values for plotting ----
    comp_time[idx] = t_opt
    success[idx] = resultFlag
    cost_plot[idx] = f_opt
    # X_plot[:, idx+1] = x0
    U_plot[:, idx] = u0
    x_opt = np.array(x_opt)
    x_opt[:, -1] += psic_offset
    X_future[:, :, idx] = np.array(x_opt)
    U_future[:, :, idx] = np.array(u_opt)
    if dyn.Nz > 0:
        del_plot[:, idx] = del_opt[:, 0].elements()
    # ---- update selection matrix ----
    if X_goal is not None and f_opt < 0.00001 and S_goal_idx > 10:
        S_goal_idx -= 1
        S_goal_val = [0]*(N_MPC-1)
        S_goal_val[S_goal_idx] = 1
        print(S_goal_val)
        # sys.exit()
#  -------------------------------------------------------------------
X_nom_val[-1, :] += psic_offset
X_plot[-1, :] += psic_offset
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)
X_pusher_opt = p_map(X_plot)
#  -------------------------------------------------------------------

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
#                 150        U_plot.transpose(),
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
    #  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
if show_anim:
    #  ---------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(
                x0_nom[:Nidx], x1_nom[:Nidx], plot_title='')
    # add computed nominal trajectory
    # X_nom_val_opt = np.array(X_nom_val_opt)
    # ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
    #         linewidth=2.0, linestyle='dashed')
    # X_nom_comp = np.array(X_nom_comp)
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
    dyn.set_patches(ax, X_plot, beta, curve.curve_func)
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
    # name the file with date and time
    file_name = './video/circle_test_' + time.strftime("%Y%m%d-%H%M%S") + '.mp4'
    ani.save(file_name, fps=25, extra_args=['-vcodec', 'mpeg4'])
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
    # axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
    #                linestyle='--', label='plan')
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
    # axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
    #                linestyle='--', label='plan')
    axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
    # if i == 1:
    #     axs[2, i].plot(t_idx_u, U_plot[2, :] - U_plot[3, :], color='green', label='dpsic')
    handles, labels = axs[2, i].get_legend_handles_labels()
    axs[2, i].legend(handles, labels)
    axs[2, i].set_xlabel('time [s]')
    axs[2, i].set_ylabel('u%d' % i)
    axs[2, i].grid()
#  -------------------------------------------------------------------

data_log = {
    'X_plot': X_plot,
    'U_plot': U_plot,
    'X_nom_val': X_nom_val,
    'X_future': X_future,
    'U_future': U_future
}

import pickle
pickle.dump(data_log, open('./data/tracking_sim_data_' + time.strftime("%Y%m%d-%H%M%S") + '.npy', 'wb'))

#  -------------------------------------------------------------------
# Save and show plots
plt.savefig('./data/tracking_exp_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
plt.show()
#  -------------------------------------------------------------------

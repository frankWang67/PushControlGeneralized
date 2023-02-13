#!/home/robotics/.conda/envs/py36_new/bin/python3
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
import sys
import yaml
import numpy as np
# import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
from utils import *
from utils.utils import *
#  -------------------------------------------------------------------
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
#  -------------------------------------------------------------------
# ----- use tf2 in python3 -----
sys.path = ['/home/roboticslab/jyp/catkin_ws/devel/lib/python3/dist-packages'] + sys.path
import rospy
import tf2_ros
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from geometry_msgs.msg import Twist
#  -------------------------------------------------------------------
# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('tracking_robot.yaml')
planning_config = sliding_pack.load_config('planning_robot.yaml')
#  -------------------------------------------------------------------

# Initialize coordinates
#  -------------------------------------------------------------------
x_axis_in_world = np.array([np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
y_axis_in_world = np.array([-np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
z_axis_in_world = np.array([0., 0., -1.])
DEFAULT_ROTATION_MATRIX = np.c_[x_axis_in_world, \
                                y_axis_in_world, \
                                z_axis_in_world]
#  -------------------------------------------------------------------

# Panda ROS control
#  -------------------------------------------------------------------
def ged_real_end_effector_xy_abs(fa:FrankaArm):
    pose = fa.get_pose()
    return pose.translation[:2]

def get_desired_end_effector_xy_abs(slider_pose, rel_coord):
    """
    :param slider_pose: the slider pose (x, y, theta)
    :param rel_pose: the relative pose (x_s, y_s) in the slider's local frama
    """
    rel_x, rel_y = rel_coord
    sx, sy, stheta = slider_pose

    rot_mat = rotation_matrix(stheta)
    abs_coord = np.array([sx, sy]) + rot_mat @ np.array([rel_x, rel_y])

    return abs_coord

def panda_move_to_pose(fa:FrankaArm, trans, trans_pre=None):
    rospy.loginfo('Moving the pusher to new goal!')

    pose_cur = fa.get_pose()
    tool_frame_name = pose_cur.from_frame
    xyz_cur = pose_cur.translation

    xyz_step1 = xyz_cur + np.array([0., 0., 0.1])
    rotmat_step1 = DEFAULT_ROTATION_MATRIX
    trans_step1 = make_rigid_transform(xyz_step1, rotmat_step1, tool_frame_name)
    fa.goto_pose(trans_step1, duration=3)

    xyz_goal = trans
    xyz_goal_pre = trans_pre

    if trans_pre is not None:
        xyz_step2 = xyz_goal_pre + np.array([0., 0., 0.1])
        rotmat_step2 = DEFAULT_ROTATION_MATRIX
        trans_step2 = make_rigid_transform(xyz_step2, rotmat_step2, tool_frame_name)
        fa.goto_pose(trans_step2, duration=5)
    else:
        xyz_step2 = xyz_goal + np.array([0., 0., 0.1])
        rotmat_step2 = DEFAULT_ROTATION_MATRIX
        trans_step2 = make_rigid_transform(xyz_step2, rotmat_step2, tool_frame_name)
        fa.goto_pose(trans_step2, duration=5)

    xyz_step3 = xyz_goal.copy()
    rotmat_step3 = DEFAULT_ROTATION_MATRIX
    trans_step3 = make_rigid_transform(xyz_step3, rotmat_step3, tool_frame_name)
    fa.goto_pose(trans_step3, duration=3)

    rospy.loginfo('Finished moving the pusher to new goal!')

    return trans_step3

def get_rel_coords_on_slider(psic, beta, contact_face, return_pre_pos):
    """
    :param return_pre_pose: if true, return the pre-contact pos
    """
    assert contact_face == 'back'
    xl, yl = beta[:2]
    if contact_face == 'back':
        rel_x = -0.5*xl
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x - 0.01
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])

    if return_pre_pos:
        return rel_coords, pre_rel_coords
    else:
        return rel_coords

# v_pub = rospy.Publisher("todo", Twist, queue_size=100)
def get_v_by_u0(u0, tf, fa:FrankaArm):
    franka_pos = fa.get_pose()
    slider_pos, slider_ori = tf.get_slider_position_and_orientation()
    vec = rotation_matrix(slider_ori).T @ (franka_pos.translation - slider_pos)[0:2] + [beta[2], 0]
    phi = np.arctan2(vec[1], vec[0])
    w = np.array([u0[0], u0[1], u0[2]-u0[3]]).reshape(3, 1)
    xC, yC = get_rel_coords_on_slider(phi, contact_face="back", return_pre_pos=False)
    JC = np.array([1, 0, -yC, 
                   0, 1, xC]).reshape(2, 3)
    GC = np.zeros(2, 3)
    L = np.diag(1, 1, 1)
    GC[:][0:2] = JC @ L @ JC.T
    GC[2] = np.array([0, -xC/(np.cos(phi)*np.cos(phi))]).reshape(2, 1)
    v_S = np.zeros(3, 1)
    v_S[0:2] = GC @ w
    trans = tf.get_transform(tf.slider_frame_name, tf.base_frame_name)
    slider_quat = np.array(trans.transform.rotation)
    slider_rotmat = R.from_quat(slider_quat).as_matrix()
    v_G = slider_rotmat @ v_S
    # ---- compute q ----
    Ja = fa.get_jacobian(fa.get_joints())
    Ja_inv = np.linalg.pinv(Ja)
    q_G = Ja_inv @ v_G
    return q_G

# ---- For Joint Velocity Control, usage similar to FrankaPyInterface
class JointVelocityControlInterface(object):
    def __init__(self, fa:FrankaArm):
        self.fa = fa
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.init_time = 0
    
    def joint_control_start(self):
        self.home_joints = self.fa.get_joints()
        max_execution_time = 30
        self.fa.dynamic_joint_velocity(joints=self.home_joints,
                              joints_vel=np.zeros((7,)),
                              duration=max_execution_time,
                              buffer_time=10,
                              block=False)
        self.init_time = rospy.Time.now().to_time()
    
    def joint_control_go(self, u0, tf):
        q_G = get_v_by_u0(u0, tf, self.fa)
        timestamp = rospy.Time.now().to_time() - self.init_time
        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=timestamp, 
            seg_run_time=30.0,
            joints=self.home_joints,
            joint_vels=np.zeros(7),
        )

        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        self.pub.publish(ros_msg)
    
    def joint_control_terminate(self):
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)
        rospy.loginfo('Done')
    
class tfHandler(object):
    def __init__(self, base_frame_name, slider_frame_name) -> None:
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.slider_frame_name = slider_frame_name
        self.base_frame_name = base_frame_name

    def get_transform(self, target_frame, source_frame):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform(target_frame, source_frame, 0)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn('The transform from {0} to {1} does not exist!'.format(source_frame, target_frame))
                rate.sleep()
                continue
        return trans

    def get_slider_position_and_orientation(self):
        """
        :return: slider_pos (x, y, z)
        :return: slider_ori theta
        """
        trans = self.get_transform(self.slider_frame_name, self.base_frame_name)
        slider_pos = np.array(trans.transform.translation)
        slider_quat = np.array(trans.transform.rotation)

        slider_rotmat = R.from_quat(slider_quat).as_matrix()
        slider_x_axis_in_world = slider_rotmat[:, 0]
        slider_ori = np.arctan2(slider_x_axis_in_world[1], slider_x_axis_in_world[0])

        return slider_pos, slider_ori

class FrankaPyInterface(object):
    def __init__(self, fa:FrankaArm) -> None:
        self.fa = fa
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    
    def pose_control_start(self, pose0):
        fa.goto_pose(pose0, duration=100, dynamic=True, buffer_time=10,
            cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        self.init_time = rospy.Time.now().to_time()

    def pose_control_goto(self, pose):
        timestamp = rospy.Time.now().to_time() - self.init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=pose.translation, quaternion=pose.quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )

        self.pub.publish(ros_msg)

    def pose_control_terminite(self):
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)

#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 10  # time of the simulation is seconds
freq = 25  # number of increments per second
# N_MPC = 12 # time horizon for the MPC controller
N_MPC = 25  # time horizon for the MPC controller
# x_init_val = [-0.03, 0.03, 30*(np.pi/180.), 0]
x_init_val = [0., 0., 45*(np.pi/180.), 0]
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
contact_point_depth = 0.04
base_frame_name = ''
slider_frame_name = ''
#  -------------------------------------------------------------------

# initialize TF2_ROS interface
#  -------------------------------------------------------------------
fa = FrankaArm(with_gripper=False)
tf_handler = tfHandler(base_frame_name=base_frame_name,
                       slider_frame_name=slider_frame_name)
#  -------------------------------------------------------------------

# initialize Franka control
#  -------------------------------------------------------------------
franka_ctrl = FrankaPyInterface(fa)
joint_velocity_ctrl = JointVelocityControlInterface(fa)
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        tracking_config['dynamics'],
        tracking_config['TO']['contactMode']
)
#  -------------------------------------------------------------------

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
X_goal = tracking_config['TO']['X_goal']
# print(X_goal)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_ellipse(-np.pi/2, 3*np.pi/2, 0.2, 0.1, N, N_MPC)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.3, N, N_MPC)
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode']
)
optObjNom = sliding_pack.to.buildOptObj(
        dynNom, N+N_MPC, planning_config['TO'], dt=dt)
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]
resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjNom.solveProblem(
        0, [0., 0., 0.*(np.pi/180.), 0.], beta,
        X_warmStart=X_nom_val)
if dyn.Nu > dynNom.Nu:
    U_nom_val_opt = cs.vertcat(
            U_nom_val_opt,
            cs.DM.zeros(np.abs(dyn.Nu - dynNom.Nu), N+N_MPC-1))
elif dynNom.Nu > dyn.Nu:
    U_nom_val_opt = U_nom_val_opt[:dyn.Nu, :]
f_d = cs.Function('f_d', [dyn.x, dyn.u], [dyn.x + dyn.f(dyn.x, dyn.u, beta)*dt])
f_rollout = f_d.mapaccum(N+N_MPC-1)
X_nom_comp = f_rollout([0., 0., 0., 0.], U_nom_val_opt)
#  ------------------------------------------------------------------

# define optimization problem
#  -------------------------------------------------------------------
optObj = sliding_pack.to.buildOptObj(
        dyn, N_MPC, tracking_config['TO'],
        X_nom_val, None, dt=dt,
)
#  -------------------------------------------------------------------

# control panda to start position
#  -------------------------------------------------------------------
slider_abs_pos, slider_abs_ori = tf_handler.get_slider_position_and_orientation()
pusher_psic_init = x_init_val[3]
x_rel_init, x_pre_rel_init = get_rel_coords_on_slider(pusher_psic_init, beta, contact_face='back', return_pre_pos=True)
panda_ee_xy_desired = get_desired_end_effector_xy_abs(np.append(slider_abs_pos[:2], slider_abs_ori), x_rel_init)
panda_ee_z_desired = slider_abs_pos[2] - contact_point_depth
panda_ee_xy_pre_desired = get_desired_end_effector_xy_abs(np.append(slider_abs_pos[:2], slider_abs_ori), x_pre_rel_init)
panda_ee_z_pre_desired = slider_abs_pos[2] - contact_point_depth
pose_init = panda_move_to_pose(fa, np.append(panda_ee_xy_desired, panda_ee_z_desired), np.append(panda_ee_xy_pre_desired, panda_ee_z_pre_desired))

rospy.loginfo('Panda moved to slider!')

joint_velocity_ctrl.joint_control_start()
#  -------------------------------------------------------------------

# Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([dyn.Nx, Nidx])
U_plot = np.empty([dyn.Nu, Nidx-1])
del_plot = np.empty([dyn.Nz, Nidx-1])
X_plot[:, 0] = x_init_val
X_future = np.empty([dyn.Nx, N_MPC, Nidx])
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
for idx in range(Nidx-1):
    print('-------------------------')
    print(idx)
    # if idx == idxDist:
    #     print('i died here')
    #     x0[0] += 0.03
    #     x0[1] += -0.03
    #     x0[2] += 30.*(np.pi/180.)
    # ---- solve problem ----
    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
            idx, x0, beta,
            S_goal_val=S_goal_val,
            obsCentre=obsCentre, obsRadius=obsRadius)
    print(f_opt)
    # ---- update initial state (simulation) ----
    u0 = u_opt[:, 0].elements()
    # ---- send joint velocity by u0 ----
    joint_velocity_ctrl.joint_control_go(u0, tf_handler)
    # x0 = x_opt[:,1].elements()
    x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()
    # ---- control Franka ----
    panda_ee_xy = ged_real_end_effector_xy_abs(fa)
    f0 = np.array([u0[0], u0[1]-u0[2]])
    # ---- store values for plotting ----
    comp_time[idx] = t_opt
    success[idx] = resultFlag
    cost_plot[idx] = f_opt
    X_plot[:, idx+1] = x0
    U_plot[:, idx] = u0
    X_future[:, :, idx] = np.array(x_opt)
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
# stop joint velocity control interface
joint_velocity_ctrl.joint_control_terminate()
#  -------------------------------------------------------------------
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(N)
X_pusher_opt = p_map(X_plot)
#  -------------------------------------------------------------------

if save_to_file:
    #  Save data to file using pandas
    #  -------------------------------------------------------------------
    # df_state = pd.DataFrame(
    #                 np.concatenate((
    #                     np.array(X_nom_val[:, :Nidx]).transpose(),
    #                     np.array(X_plot).transpose(),
    #                     np.array(X_pusher_opt).transpose()
    #                     ), axis=1),
    #                 columns=['x_nom', 'y_nom', 'theta_nom', 'psi_nom',
    #                          'x_opt', 'y_opt', 'theta_opt', 'psi_opt',
    #                          'x_pusher', 'y_pusher'])
    # df_state.index.name = 'idx'
    # df_state.to_csv('tracking_circle_cc_state.csv',
    #                 float_format='%.5f')
    time = np.linspace(0., T, Nidx-1)
    print('********************')
    print(U_plot.transpose().shape)
    print(cost_plot.shape)
    print(comp_time.shape)
    print(time.shape)
    print(time[:, None].shape)
    # df_action = pd.DataFrame(
    #                 np.concatenate((
    #                     U_plot.transpose(),
    #                     time[:, None],
    #                     cost_plot,
    #                     comp_time
    #                     ), axis=1),
    #                 columns=['u0', 'u1', 'u3', 'u4',
    #                 # columns=['u0', 'u1', 'u3',
    #                          'time', 'cost', 'comp_time'])
    # df_action.index.name = 'idx'
    # df_action.to_csv('tracking_circle_cc_action.csv',
    #                  float_format='%.5f')
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

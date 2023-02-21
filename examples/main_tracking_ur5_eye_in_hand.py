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
import pickle
import time
# import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
from sliding_pack.utils import *
from sliding_pack.utils.utils import *
#  -------------------------------------------------------------------
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
#  -------------------------------------------------------------------
# ----- use tf2 in python3 -----
sys.path = ['/home/roboticslab/jyp/catkin_ws/devel/lib/python3/dist-packages'] + sys.path
import rospy
import tf2_ros
from std_srvs.srv import Trigger
from aruco_ros.srv import AskForMarkerCenter
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from geometry_msgs.msg import Twist, TransformStamped
#  -------------------------------------------------------------------
# Get config files
#  -------------------------------------------------------------------
tracking_config = sliding_pack.load_config('tracking_robot.yaml')
planning_config = sliding_pack.load_config('planning_robot.yaml')
beta = [
    planning_config['dynamics']['xLenght'],
    planning_config['dynamics']['yLenght'],
    planning_config['dynamics']['pusherRadious']
]
#  -------------------------------------------------------------------

# Initialize coordinates
#  -------------------------------------------------------------------
x_axis_in_world = np.array([-np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
y_axis_in_world = np.array([-np.sqrt(2)/2., np.sqrt(2)/2., 0.])
z_axis_in_world = np.array([0., 0., -1.])
DEFAULT_ROTATION_MATRIX = np.c_[x_axis_in_world, \
                                y_axis_in_world, \
                                z_axis_in_world]
GOD_VIEW_XYZ_RVEC = [0.4335328270310983, 0.4960794911382269, -0.1233926796537342, -0.690356196985449, 1.753462137271583, 1.8056789286362698]
T_CAMERA2UREE = RigidTransform(translation=np.array([0.09370675903545315, -0.13205444680184125, -0.2692020808263898]),
                               rotation=R.from_quat(np.array([-0.19109339366918357, -0.09436200160334807, 0.3927029150218745, 0.8946303974730163])).as_matrix(),
                               from_frame='camera_link', to_frame='ur_ee')
T_URBLINK2WORLD = RigidTransform(translation=np.array([0, 0.357, 0]),
                                 rotation=R.from_quat(np.array([-4.32978028e-17, 7.07106781e-01, 7.07106781e-01, 4.32978028e-17])).as_matrix(),
                                 from_frame='base_link', to_frame='world')
T_URBASE2URBLINK = RigidTransform(translation=np.array([0, 0, 0]),
                                 rotation=R.from_quat(np.array([0, 0, 1, 0])).as_matrix(),
                                 from_frame='base', to_frame='base_link')
T_URBASE2WORLD = T_URBLINK2WORLD * T_URBASE2URBLINK
#  -------------------------------------------------------------------

# Panda ROS control
#  -------------------------------------------------------------------
def rtrans2rvecpose(pose: RigidTransform):
    pose_xyz = pose.translation.tolist()
    pose_rvec = R.from_matrix(pose.rotation).as_rotvec().tolist()
    pose_xyz_rvec = pose_xyz + pose_rvec
    return pose_xyz_rvec

def rvecpose2rtrans(pose):
    pose_rtrans = RigidTransform()
    pose_rtrans.translation = pose[0:3]
    pose_rtrans.rotation = R.from_rotvec(pose[3:6]).as_matrix()
    return pose_rtrans

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

def get_rel_coords_on_slider(psic, beta, contact_face, return_pre_pos):
    """
    :param return_pre_pose: if true, return the pre-contact pos
    """
    xl, yl, rl = beta
    if contact_face == 'back':
        rel_x = -0.5*xl - rl
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x - 0.02
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'front':
        rel_x = 0.5*xl + rl
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x + 0.02
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'left':
        rel_y = 0.5*yl + rl
        rel_x = -rel_y * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x
            pre_rel_y = rel_y + 0.02
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'right':
        rel_y = -0.5*yl - rl
        rel_x = -rel_y * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x
            pre_rel_y = rel_y - 0.02
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    else:
        raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))

    if return_pre_pos:
        return rel_coords, pre_rel_coords
    else:
        return rel_coords

def get_contact_point_on_slider(psic, beta, contact_face):
    """
    :param return_pre_pose: if true, return the pre-contact pos
    """
    xl, yl, _ = beta
    if contact_face == 'back':
        rel_x = -0.5*xl
        rel_y = rel_x * np.tan(psic)
    elif contact_face == 'front':
        rel_x = 0.5*xl
        rel_y = rel_x * np.tan(psic)
    elif contact_face == 'left':
        rel_y = 0.5*yl
        rel_x = -rel_y * np.tan(psic)
    elif contact_face == 'right':
        rel_y = -0.5*yl
        rel_x = -rel_y * np.tan(psic)
    else:
        raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))

    contact_point = np.array([rel_x, rel_y])
    return contact_point

def get_abs_coords_on_slider(rel_coords, slider_pose):
    theta = slider_pose[2]
    xy_pos = slider_pose[:2]
    rot_mat = rotation_matrix(theta)
    abs_coords = xy_pos + rot_mat @ rel_coords

    return abs_coords

class tfHandler(object):
    def __init__(self, base_frame_name, slider_frame_name, ur) -> None:
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.ur = ur

        self.slider_frame_name = slider_frame_name
        self.base_frame_name = base_frame_name

    def publish_ur5_ee_tf(self, pose:RigidTransform):
        tf_br = tf2_ros.TransformBroadcaster()
        tf_msg = TransformStamped()

        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "base"
        tf_msg.child_frame_id = "ur5_EE"
        tf_msg.transform.translation.x = pose.translation[0]
        tf_msg.transform.translation.y = pose.translation[1]
        tf_msg.transform.translation.z = pose.translation[2]
        quat = R.from_matrix(pose.rotation).as_quat()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        tf_br.sendTransform(tf_msg)

    def get_transform(self, target_frame, source_frame, return_rigid_transform):
        # self.publish_ur5_ee_tf(self.ur.get_pose(in_world=False))
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform("camera_link", "marker1_frame", rospy.Time(0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('The transform from {0} to {1} does not exist!'.format("marker1_frame", "camera_link"))
                rospy.logwarn('The transform from {0} to {1} does not exist!'.format("marker1_frame", "camera_link"))
                rate.sleep()
                continue
        
        T_SLIDER2CAM = RigidTransform(from_frame="marker1_frame", to_frame="camera_link")
        T_SLIDER2CAM.translation = np.array([trans.transform.translation.x, \
                                            trans.transform.translation.y, \
                                            trans.transform.translation.z])
        T_SLIDER2CAM.rotation = R.from_quat(np.array([trans.transform.rotation.x, \
                                                        trans.transform.rotation.y, \
                                                        trans.transform.rotation.z, \
                                                        trans.transform.rotation.w])).as_matrix()

        T_UREE2BASE = self.ur.get_pose(in_world=False)
        T_SLIDER2WORLD = T_URBASE2WORLD * T_UREE2BASE * T_CAMERA2UREE * T_SLIDER2CAM
        
        if return_rigid_transform:
            return T_SLIDER2WORLD
        else:
            trans = TransformStamped()
            trans.transform.translation.x = T_SLIDER2WORLD.translation[0]
            trans.transform.translation.y = T_SLIDER2WORLD.translation[1]
            trans.transform.translation.z = T_SLIDER2WORLD.translation[2]
            quat = R.from_matrix(T_SLIDER2WORLD.rotation).as_quat().reshape(-1)
            trans.transform.rotation.x = quat[0]
            trans.transform.rotation.y = quat[1]
            trans.transform.rotation.z = quat[2]
            trans.transform.rotation.w = quat[3]
            return trans

    def get_slider_position_and_orientation(self):
        """
        :return: slider_pos (x, y, z)
        :return: slider_ori theta
        """
        trans = self.get_transform(self.base_frame_name, self.slider_frame_name, return_rigid_transform=False)
        slider_pos = np.array([trans.transform.translation.x, \
                               trans.transform.translation.y, \
                               trans.transform.translation.z])
        slider_quat = np.array([trans.transform.rotation.x, \
                                trans.transform.rotation.y, \
                                trans.transform.rotation.z, \
                                trans.transform.rotation.w])

        slider_rotmat = R.from_quat(slider_quat).as_matrix()
        slider_x_axis_in_world = slider_rotmat[:, 0]
        slider_ori = np.arctan2(slider_x_axis_in_world[1], slider_x_axis_in_world[0])

        return slider_pos, slider_ori

class UR5Arm(object):
    def __init__(self, robot_ip) -> None:
        self.receive = RTDEReceive(robot_ip)
        self.control = RTDEControl(robot_ip)

    def set_base_in_world(self, base_in_world:RigidTransform):
        self.base_in_world = base_in_world

    def get_pose(self, in_world=True):
        pose_xyz_rvec = self.receive.getActualTCPPose()
        pose_rtrans = RigidTransform(from_frame='ur_ee', to_frame='base')
        pose_rtrans.translation = pose_xyz_rvec[0:3]
        pose_rtrans.rotation = R.from_rotvec(pose_xyz_rvec[3:6]).as_matrix()

        if in_world:
            pose_in_base = copy.deepcopy(pose_rtrans)
            pose_in_world = self.base_in_world * pose_in_base
            return pose_in_world
        else:
            pose_in_base = copy.deepcopy(pose_rtrans)
            return pose_in_base

    def get_velocity_ee(self, in_world=False):
        speed_vec_base = self.receive.getActualTCPSpeed()
        if in_world:
            speed_vec_world = self.base_in_world.rotation @ speed_vec_base[0:3]
            return [speed_vec_world[0], speed_vec_world[1]]
        else:
            return [speed_vec_base[0], speed_vec_base[1]]

    def apply_velocity_ee(self, v_xy, in_world=True):
        if in_world:
            v_xyz_world = np.append(v_xy, 0.)
            v_xyz_base = self.base_in_world.rotation.T @ v_xyz_world
        else:
            v_xyz_base = np.append(v_xy, 0.)
        v_spatial_vec = np.append(v_xyz_base, [0., 0., 0.]).tolist()
        success = self.control.speedL(v_spatial_vec, time=0)
        return success

    def goto_pose(self, pose:RigidTransform, use_world_frame=True, speed=0.1, asynchronous=False):
        if use_world_frame:
            pose_in_world = copy.deepcopy(pose)
            pose_in_base = self.base_in_world.inverse() * pose_in_world
        else:
            pose_in_base = copy.deepcopy(pose)
        pose_xyz = pose_in_base.translation.tolist()
        pose_rvec = R.from_matrix(pose_in_base.rotation).as_rotvec().tolist()
        pose_xyz_rvec = pose_xyz + pose_rvec
        success = self.control.moveL(pose_xyz_rvec, speed=speed, acceleration=0.6, asynchronous=asynchronous)
        return success

# ---- For Joint Velocity Control, usage similar to FrankaPyInterface
class JointVelocityControlInterface(object):
    def __init__(self, ur:UR5Arm, tf:tfHandler, beta, dt):
        self.ur = ur
        self.tf = tf
        self.L_surf = None
        self.beta = beta
        self.dt = dt

        self.init_time = 0
        self.i = 0

        # record data
        self.record = []
        self.vel_desired = []
        self.vel_actual = []

        self.next_xy_desired = []
        self.actual_xy_executed = []
        self.x_dot_desired = []  # dyn.f(x0, u0, beta)

        self.v_G_lpf_coeff = 0.05
        self.last_v_G = None

    def set_L_surf(self, L_surf):
        self.L_surf = L_surf

    def get_psic(self):
        ur_pose = self.ur.get_pose()
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        vec = rotation_matrix(slider_ori).T @ (ur_pose.translation - slider_pos)[0:2] + [self.beta[2], 0]
        phi = np.arctan2(-vec[1], -vec[0])
        return phi

    def get_contact_jacobian(self, psic, contact_face):
        xc, yc = get_contact_point_on_slider(psic, self.beta, contact_face)
        Jc = np.array([[1, 0, -yc],
                       [0, 1, xc]])
        return Jc

    def ur5_goto_god_view(self):
        """
        Control ur5 to god view
        """
        # lift up to detach the slider
        pose_cur = self.ur.get_pose()
        
        pose_liftup = pose_cur.copy()
        if pose_liftup.translation[2] < -0.55:
            pose_liftup.translation[2] += 2 * contact_point_depth
            self.ur.goto_pose(pose_liftup)

        pos_xyz_rvec_home = GOD_VIEW_XYZ_RVEC
        pose_rtrans = rvecpose2rtrans(pos_xyz_rvec_home)
        self.ur.goto_pose(pose_rtrans, use_world_frame=False)

    def get_ee_pos_in_contact(self, contact_face, get_pre_contact_pos=True):
        """
        Get the ee position when in contact
        """
        contact_ee_rel_pos, pre_contact_ee_rel_pos = get_rel_coords_on_slider(0., beta, contact_face, return_pre_pos=True)
        if get_pre_contact_pos:
            return pre_contact_ee_rel_pos
        else:
            return contact_ee_rel_pos

    def get_slider_transform_coarse(self):
        """
        Get coarse slider pose
        """
        self.ur5_goto_god_view()
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def get_slider_transform_fine(self):
        """
        Get finer slider pose
        """
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def goto_face_coarse(self, contact_face):
        """
        Go to contact, based on coarse pose measurement
        """
        slider_state, slider_height = self.get_slider_transform_coarse()
        ee_rel_pos = self.get_ee_pos_in_contact(contact_face, get_pre_contact_pos=True)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = self.ur.get_pose()
        
        ee_pose_via_point = ee_pose_cur.copy()
        ee_pose_via_point.translation[0:2] = ee_abs_xy
        ee_pose_via_point.rotation = DEFAULT_ROTATION_MATRIX
        self.ur.goto_pose(ee_pose_via_point)

        ee_pose_goal_point = ee_pose_via_point.copy()
        ee_pose_goal_point.translation[2] = slider_height - contact_point_depth
        ee_pose_goal_point.rotation = DEFAULT_ROTATION_MATRIX
        self.ur.goto_pose(ee_pose_goal_point)
        print('Info: franka arrived at slider coarsely!')

    def goto_face_fine(self, contact_face):
        """
        Go to contact, based on finer pose measurement
        """
        slider_state, slider_height = self.get_slider_transform_fine()
        ee_rel_pos = self.get_ee_pos_in_contact(contact_face, get_pre_contact_pos=False)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = self.ur.get_pose()

        ee_pose_goal_point = ee_pose_cur.copy()
        ee_pose_goal_point.translation[0:2] = ee_abs_xy
        ee_pose_goal_point.translation[2] = slider_height - contact_point_depth
        ee_pose_goal_point.rotation = DEFAULT_ROTATION_MATRIX

        self.ur.goto_pose(ee_pose_goal_point)
        print('Info: franka arrived at slider finely!')

    def change_contact_face(self, contact_face):
        """
        Change another contact face
        """
        print('Info: change contact face!')
        self.goto_face_coarse(contact_face)
        self.goto_face_fine(contact_face)
        self.pusher_tip_height = self.ur.get_pose().translation[2]

    def velocity_control_go(self, u, contact_face):
        psic = self.get_psic()
        u_extend = [u[0], u[1], u[2]-u[3]]
        Jc = self.get_contact_jacobian(psic, contact_face)
        Gc = np.zeros((2, 3))
        Gc[:, 0:2] = (Jc @ self.L_surf @ Jc.T)
        xc = Jc[1, 2]
        Gc[:, 2] = np.array([0, xc/(np.cos(psic)*np.cos(psic))])

        v_desired = Gc @ u_extend  # in slider frame
        _, slider_theta = tf_handler.get_slider_position_and_orientation()
        rot_mat = rotation_matrix(slider_theta)

        v_desired = (rot_mat @ v_desired) / VELOCITY_SCALE  # in world frame
        if np.linalg.norm(v_desired) >= MAX_VELOCITY_DESIRED:
            v_desired = MAX_VELOCITY_DESIRED * v_desired / np.linalg.norm(v_desired)

        # ------------------------------------------------
        self.vel_desired.append(v_desired.tolist())
        self.vel_actual.append(self.ur.get_velocity_ee(in_world=True))
        # ------------------------------------------------

        assert len(v_desired.reshape(-1).tolist()) == 2
        print(v_desired)
        success = self.ur.apply_velocity_ee(v_desired)
        if not success:
            rospy.logwarn('Failed to control UR5 with velocity {0}!'.format(v_desired))

    def velocity_control_stop(self):
        success = self.ur.apply_velocity_ee([0., 0.])
        if not success:
            rospy.logwarn('Failed to control UR5 with velocity {0}!'.format([0., 0.]))

    def pose_control_go(self, x1):
        xy_ee_next = get_abs_coords_on_slider(
                        get_rel_coords_on_slider(x1[3], beta, contact_face='back', return_pre_pos=False),
                        x1[:3]
                    )

        # ------------------------------------------------
        self.actual_xy_executed.append(self.ur.get_pose().translation[0:2].tolist())
        self.next_xy_desired.append(xy_ee_next[0:2])
        # ------------------------------------------------

        pose_cur = self.ur.get_pose()
        xy_ee_now = pose_cur.translation[0:2]

        speed_desired = np.linalg.norm(xy_ee_next-xy_ee_now) / self.dt

        pose = RigidTransform(translation=np.append(xy_ee_next, self.pusher_tip_height), rotation=DEFAULT_ROTATION_MATRIX, \
                              from_frame=pose_cur.from_frame, to_frame=pose_cur.to_frame)

        if speed_desired > MAX_VELOCITY_DESIRED:
            rospy.logwarn('Maximum velocity limit exceeded, ur5 will stop move!')
            pose = pose_cur.copy()

        success = self.ur.goto_pose(pose, speed=speed_desired, asynchronous=True)
        if not success:
            rospy.logwarn('Failed to control UR5 from {0} to {1}!'.format(xy_ee_now, xy_ee_next))

#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 8  # time of the simulation is seconds
freq = 25  # number of increments per second
# N_MPC = 12 # time horizon for the MPC controller
N_MPC = 30  # time horizon for the MPC controller
# x_init_val = [-0.03, 0.03, 30*(np.pi/180.), 0]
# x_init_val = [0., 0., 45*(np.pi/180.), 0]
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
contact_point_depth = 0.02
L_SURF_SCALE = 1.3888888888888888
# L_SURF_SCALE = 1.
VELOCITY_SCALE = 10.
MAX_VELOCITY_DESIRED = 1.0
MAX_TRANSLATION_DESIRED = 0.05  # the maximum translation when running (dynamic) goto_pose

LARGER_TRANSLATIONAL_STIFFNESS = [2400.0, 2400.0, 2400.0]
LARGER_ROTATIONAL_STIFFNESS = [50.0, 50.0, 50.0]

## disturbance observer gain
K_d = 0.0  # set K_d>0 to enable DOB

ur_robot_ip = '192.168.101.50'
base_frame_name = 'world'
ur5_base_frame_name = 'base'
slider_frame_name = 'marker1_frame'
default_contact_face = 'back'

# controller_type = 'moveL'
controller_type = 'speedL'
#  -------------------------------------------------------------------
nh = rospy.init_node('pusher_slider_controller', anonymous=True)
# initialize TF2_ROS interface
#  -------------------------------------------------------------------
ur = UR5Arm(robot_ip=ur_robot_ip)
tf_handler = tfHandler(base_frame_name=base_frame_name,
                       slider_frame_name=slider_frame_name,
                       ur=ur)
T_urbase2world = T_URBASE2WORLD
ur.set_base_in_world(T_urbase2world)
#  -------------------------------------------------------------------

# initialize Franka control
#  -------------------------------------------------------------------
joint_velocity_ctrl = JointVelocityControlInterface(ur, tf_handler, beta, dt)
#  -------------------------------------------------------------------

# initialize Franka pose
#  -------------------------------------------------------------------
import pdb; pdb.set_trace()
joint_velocity_ctrl.change_contact_face('back')
rospy.loginfo('Move the pusher to start position!')
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        tracking_config['dynamics'],
        tracking_config['TO']['contactMode'],
        pusherAngleLim=tracking_config['dynamics']['xFacePsiLimit'],
        limit_surf_gain=1.
)
#  -------------------------------------------------------------------

# get slider init pos
slider_abs_pos, slider_abs_ori = tf_handler.get_slider_position_and_orientation()
# set x_init
x_init_val = [slider_abs_pos[0], slider_abs_pos[1], slider_abs_ori, 0.]

# Generate Nominal Trajectory
#  -------------------------------------------------------------------
X_goal = tracking_config['TO']['X_goal']
# print(X_goal)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(X_goal[0], X_goal[1], N, N_MPC)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.05, -0.05, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.0, -0.35, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.0, -0.35, 0.04, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_ellipse(-np.pi/2, 3*np.pi/2, 0.2, 0.1, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.3, N, N_MPC)

# test short trajectory
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.05, 0.0, N, N_MPC)

# test self defined trajectory
# import pickle
# path_seg = pickle.load(open('/home/roboticslab/jyp/pusher_slider/data/traj/path_seg_real_case1.pkl', 'rb'))
# x_slider = np.array(path_seg[0]['X_slider'])
# x0_nom, x1_nom = x_slider[:, 0], x_slider[:, 1]
# x2_nom = x_slider[:, 2]
# x0_nom = sliding_pack.utils.interpolate_path(x0_nom, 88)
# x1_nom = sliding_pack.utils.interpolate_path(x1_nom, 88)
# x2_nom = sliding_pack.utils.interpolate_path(x2_nom, 88)
# T = 3.5; N = int(round(T * freq)); Nidx = N * 4
# x0_nom_mpc = x0_nom[-1] + np.repeat((x0_nom[-1]-x0_nom[-2]), N_MPC).cumsum()
# x0_nom = np.concatenate((x0_nom, x0_nom_mpc))
# x1_nom_mpc = x1_nom[-1] + np.repeat((x1_nom[-1]-x1_nom[-2]), N_MPC).cumsum()
# x1_nom = np.concatenate((x1_nom, x1_nom_mpc))
# x2_nom_mpc = x2_nom[-1] + np.repeat((x2_nom[-1]-x2_nom[-2]), N_MPC).cumsum()
# x2_nom = np.concatenate((x2_nom, x2_nom_mpc))
# x0_nom -= x0_nom[0]
# x1_nom -= x1_nom[0]

# add offset to trajectory
#  -------------------------------------------------------------------
"""
nom_traj_offset = [0.03, -0.03]
x0_nom += nom_traj_offset[0]
x1_nom += nom_traj_offset[1]
"""

x0_nom += x_init_val[0]
x1_nom += x_init_val[1]
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# X_nom_val[2, :] = x2_nom

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode'],
        pusherAngleLim=tracking_config['dynamics']['xFacePsiLimit']
)
optObjNom = sliding_pack.to.buildOptObj(
        dynNom, N+N_MPC, planning_config['TO'], dt=dt, max_iter=100)

# initialize limit surface matrix
L_surf = dyn.A(beta).toarray() * L_SURF_SCALE
joint_velocity_ctrl.set_L_surf(L_surf)

print('Limit surface ready!')

resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjNom.solveProblem(
        0, [0., 0., 0.*(np.pi/180.), 0.], beta, [0.]*dynNom.Nx,
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
        X_nom_val, None, dt=dt, max_iter=None
)
#  -------------------------------------------------------------------

# control panda to start position
#  -------------------------------------------------------------------
# pusher_psic_init = x_init_val[3]
# x_rel_init, x_pre_rel_init = get_rel_coords_on_slider(pusher_psic_init, beta, contact_face='back', return_pre_pos=True)
# x_rel_init = x_rel_init + [0.0, 0.0]
# panda_ee_xy_desired = get_desired_end_effector_xy_abs(np.append(slider_abs_pos[:2], slider_abs_ori), x_rel_init)
# panda_ee_z_desired = slider_abs_pos[2] - contact_point_depth
# panda_ee_xy_pre_desired = get_desired_end_effector_xy_abs(np.append(slider_abs_pos[:2], slider_abs_ori), x_pre_rel_init)
# panda_ee_z_pre_desired = slider_abs_pos[2] - contact_point_depth
# pose_init = panda_move_to_pose(fa, np.append(panda_ee_xy_desired, panda_ee_z_desired), np.append(panda_ee_xy_pre_desired, panda_ee_z_pre_desired))
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

# Disturbance
#  -------------------------------------------------------------------
"""
D_hat_plot = np.empty([dyn.Nx, Nidx])  # observed disturbance
D_true_plot = np.empty([dyn.Nx, Nidx])  # actual disturbance
"""
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
import pdb; pdb.set_trace()
rosrate = rospy.Rate(freq)
x0 = x_init_val
# last_state = x0.copy()

## initialize disturbance
d0 = [0.] * dyn.Nx
"""
D_hat_plot[:, 0] = d0
D_true_plot[:, 0] = [0.]*dyn.Nx
x1_estimation = None
"""

d_hat = d0.copy()

# Nidx = X_nom_val.shape[1]
for idx in range(Nidx-1):
    time_loop_start = time.time()
    print('-------------------------')
    print(idx)
    # if idx == idxDist:
    #     print('i died here')
    #     x0[0] += 0.03
    #     x0[1] += -0.03
    #     x0[2] += 30.*(np.pi/180.)

    # ---- solve problem ----
    # make sure the slider marker is in view
    """
    if not check_slider_marker_in_view():
        if controller_type == 'joint_velocity':
            joint_velocity_ctrl.joint_control_stop_move()
            rospy.sleep(1.0)
            joint_velocity_ctrl.joint_control_terminate()
            joint_velocity_ctrl.refind_slider_out_of_view()
            import pdb; pdb.set_trace()
            joint_velocity_ctrl.joint_control_start()
        elif controller_type == 'ee_pose':
            joint_velocity_ctrl.pose_control_stop_move()
            rospy.sleep(1.0)
            joint_velocity_ctrl.pose_control_terminite()
            joint_velocity_ctrl.refind_slider_out_of_view()
            import pdb; pdb.set_trace()
            joint_velocity_ctrl.pose_control_start()
    """

    slider_abs_pos, slider_abs_ori = tf_handler.get_slider_position_and_orientation()
    psic = joint_velocity_ctrl.get_psic()
    x0 = [slider_abs_pos[0], slider_abs_pos[1], slider_abs_ori, psic]
    
    ## update disturbance estimation
    """
    if x1_estimation is not None:
        d_hat = (np.array(d_hat)+K_d*(np.array(x0) - np.array(x1_estimation)) / dt).tolist()
    D_hat_plot[:, idx+1] = d_hat
    D_true_plot[:, idx+1] = np.zeros(dyn.Nx)
    """

    # add noise to theta
    # x0[2] = x0[2] + np.random.uniform(low=-0.1, high=0.1)
    # add low path filter to theta
    # x0 = (np.array(x0) * (1 - 0.75) + np.array(last_state) * 0.75).tolist()
    # last_state = x0.copy()

    X_plot[:, idx+1] = x0

    ## find the nearest nominal point
    # nearest_idx = np.argmin(np.linalg.norm(X_nom_val[0:2, :-N_MPC] - x0[0:2], axis=0))
    nearest_idx = idx

    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
            nearest_idx, x0, beta, d_hat,
            S_goal_val=S_goal_val,
            obsCentre=obsCentre, obsRadius=obsRadius)
    print('>>> idx:{0}, x0 for opt: {1}, f_opt:{2}'.format(nearest_idx, x0, f_opt))
    # ---- update initial state (simulation) ----
    u0 = u_opt[:, 0].elements()
    # x1_estimation = x_opt[:, 1].elements()

    # ---- send joint velocity by u0 (v0) ----
    if controller_type == 'moveL':
        x1 = (x0 + (dyn.f(x0, u0, beta) + d0)*dt).elements()
        joint_velocity_ctrl.pose_control_go(x1)
    elif controller_type == 'speedL':
        joint_velocity_ctrl.velocity_control_go(u0, contact_face=default_contact_face)
    joint_velocity_ctrl.x_dot_desired.append((dyn.f(x0, u0, beta)*dt).elements())
    # ---- send joint velocity by u0 (v1) ----
    # x0_old = x0.copy()
    # x0 = (x0 + dyn.f(x0, u0, beta)*dt).elements()
    # joint_velocity_ctrl.joint_control_go_delta_x(x0_old, x0.copy(), dt, tf_handler)

    # x0 = x_opt[:,1].elements()
    # ---- control Franka ----
    # panda_ee_xy = ged_real_end_effector_xy_abs(fa)
    # f0 = np.array([u0[0], u0[1]-u0[2]])
    # ---- store values for plotting ----
    comp_time[idx] = t_opt
    success[idx] = resultFlag
    cost_plot[idx] = f_opt
    U_plot[:, idx] = u0
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
    rosrate.sleep()
    time_loop_end = time.time()
    print('loop time cost: ', time_loop_end - time_loop_start)

regular_tracking_phase_last_idx = idx

## Uncomment to allow the convergence phase
## More time steps until convergence to goal
#  -------------------------------------------------------------------
converge_phase_u_list = []
while True:
    idx = idx + 1
    Nidx = Nidx + 1

    time_loop_start = time.time()
    print('-------------------------')
    print(idx)

    slider_abs_pos, slider_abs_ori = tf_handler.get_slider_position_and_orientation()
    psic = joint_velocity_ctrl.get_psic()
    x0 = [slider_abs_pos[0], slider_abs_pos[1], slider_abs_ori, psic]

    X_plot = np.concatenate((X_plot, np.expand_dims(x0, axis=1)), axis=1)

    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(
            regular_tracking_phase_last_idx, x0, beta, d_hat,
            S_goal_val=S_goal_val,
            obsCentre=obsCentre, obsRadius=obsRadius)
    print('>>> idx:{0}, x0 for opt: {1}, f_opt:{2}'.format(regular_tracking_phase_last_idx, x0, f_opt))

    u0 = u_opt[:, 0].elements()

    if controller_type == 'moveL':
        x1 = (x0 + (dyn.f(x0, u0, beta) + d0)*dt).elements()
        joint_velocity_ctrl.pose_control_go(x1)
    elif controller_type == 'speedL':
        joint_velocity_ctrl.velocity_control_go(u0, contact_face=default_contact_face)
    joint_velocity_ctrl.x_dot_desired.append((dyn.f(x0, u0, beta)*dt).elements())

    comp_time = np.concatenate((comp_time, np.expand_dims([t_opt], axis=1)), axis=0)
    success = np.append(success, resultFlag)
    cost_plot = np.concatenate((cost_plot, np.expand_dims([f_opt], axis=1)), axis=0)
    U_plot = np.concatenate((U_plot, np.expand_dims(u0, axis=1)), axis=1)
    X_future = np.concatenate((X_future, np.expand_dims(x_opt, axis=2)), axis=2)
    if dyn.Nz > 0:
        del_plot = np.concatenate((del_plot, np.expand_dims(del_opt[:, 0].elements(), axis=1)), axis=1)
    # ---- update selection matrix ----
    if X_goal is not None and f_opt < 0.00001 and S_goal_idx > 10:
        S_goal_idx -= 1
        S_goal_val = [0]*(N_MPC-1)
        S_goal_val[S_goal_idx] = 1
        print(S_goal_val)
        # sys.exit()
    rosrate.sleep()
    time_loop_end = time.time()
    print('loop time cost: ', time_loop_end - time_loop_start)

    # detect convergence
    if len(converge_phase_u_list) >= 10:
        converge_phase_u_list.pop(0)
    converge_phase_u_list.append(u0)
    if len(converge_phase_u_list) >= 10:
        max_force_recent_applied = np.linalg.norm(np.array(converge_phase_u_list)[:, 0:2], axis=1).max()
        if idx > 600:
            break
        else:
            print('Convergence phase: max force applied recently: ', max_force_recent_applied)

# stop the robot
#  -------------------------------------------------------------------
if controller_type == 'speedL':
    joint_velocity_ctrl.velocity_control_stop()
#  -------------------------------------------------------------------
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
p_new = cs.Function('p_new', [dyn.x], [dyn.p(dyn.x, beta)])
p_map = p_new.map(Nidx)
X_pusher_opt = p_map(X_plot)
#  -------------------------------------------------------------------

"""
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
"""

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
    ani.save('./videos/MPC_MPCC_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 4, sharex=True)
fig.set_size_inches(10, 10, forward=True)
t_Nx = np.linspace(0, T, Nidx)
t_Nu = np.linspace(0, T, Nidx-1)
t_idx_x = t_Nx[0:Nidx]
t_idx_u = t_Nx[0:Nidx-1]
ctrl_g_idx = dyn.g_u.map(Nidx-1)
ctrl_g_val = ctrl_g_idx(U_plot, del_plot)
# #  -------------------------------------------------------------------
# # plot position
for i in range(dyn.Nx):
    axs[0, i].plot(t_Nx, X_nom_val[i, 0:Nidx].T, color='red',
                   linestyle='--', label='nom')
    axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:Nidx].T, color='blue',
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
# # plot actions
for i in range(dyn.Nu):
    axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:Nidx-1].T, color='blue',
                   linestyle='--', label='plan')
    axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
    handles, labels = axs[2, i].get_legend_handles_labels()
    axs[2, i].legend(handles, labels)
    axs[2, i].set_xlabel('time [s]')
    axs[2, i].set_ylabel('u%d' % i)
    axs[2, i].grid()
#  -------------------------------------------------------------------
# plot joint velocity
fig = plt.figure("velocity")
axs_x = fig.add_subplot(1, 2 ,1)
v_Gs = joint_velocity_ctrl.record
v_x = [x[0] for x in v_Gs]
v_y = [x[1] for x in v_Gs]
axs_x.plot(list(range(len(v_x))), v_x)
axs_y = fig.add_subplot(1, 2 ,2)
axs_y.plot(list(range(len(v_y))), v_y)
#  -------------------------------------------------------------------

## out own data plot
fig, axs = plt.subplots(4, 4, sharex=True)

axs[0, 0].plot(t_idx_x, X_plot[2, :].T, label='theta')
axs[0, 0].plot(t_idx_x, X_plot[3, :].T, label='psic')
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].legend(handles, labels)
axs[0, 0].set_xlabel('time [s]')
axs[0, 0].set_ylabel('theta & psic')
axs[0, 0].grid()

axs[0, 1].plot(t_idx_u, U_plot[0, :].T, label='fn')
axs[0, 1].plot(t_idx_u, U_plot[1, :].T, label='ft')
axs[0, 1].plot(t_idx_u, (U_plot[2, :]-U_plot[3, :]).T, label='dpsic')
handles, labels = axs[0, 1].get_legend_handles_labels()
axs[0, 1].legend(handles, labels)
axs[0, 1].set_xlabel('time [s]')
axs[0, 1].set_ylabel('force and dpsic')
axs[0, 1].grid()

"""
for i in range(dyn.Nx):
    axs[1, i].plot(t_Nx, D_hat_plot[i, 0:N].T, color='red',
                   linestyle='--', label='pred')
    axs[1, i].plot(t_Nx, D_true_plot[i, 0:N].T, color='blue',
                   linestyle='--', label='real')
    handles, labels = axs[1, i].get_legend_handles_labels()
    axs[1, i].legend(handles, labels)
    axs[1, i].set_xlabel('time [s]')
    axs[1, i].set_ylabel('d_hat_{}'.format(i))
    axs[1, i].grid()
"""

# plot debug: desired and actual ee xy
if controller_type == 'moveL':
    axs[2, 0].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.next_xy_desired)[:-1, 0], color='red',
                linestyle='--', label='desired')
    axs[2, 0].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.actual_xy_executed)[1:, 0], color='blue',
                linestyle='--', label='actual')
    handles, labels = axs[2, 0].get_legend_handles_labels()
    axs[2, 0].legend(handles, labels)
    axs[2, 0].set_xlabel('time [s]')
    axs[2, 0].set_ylabel('ee_x')
    axs[2, 0].grid()

    axs[2, 1].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.next_xy_desired)[:-1, 1], color='red',
                linestyle='--', label='desired')
    axs[2, 1].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.actual_xy_executed)[1:, 1], color='blue',
                linestyle='--', label='actual')
    handles, labels = axs[2, 1].get_legend_handles_labels()
    axs[2, 1].legend(handles, labels)
    axs[2, 1].set_xlabel('time [s]')
    axs[2, 1].set_ylabel('ee_y')
    axs[2, 1].grid()
elif controller_type == 'speedL':
    axs[2, 0].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.vel_desired)[:-1, 0], color='red',
                linestyle='--', label='desired')
    axs[2, 0].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.vel_actual)[1:, 0], color='blue',
                linestyle='--', label='actual')
    handles, labels = axs[2, 0].get_legend_handles_labels()
    axs[2, 0].legend(handles, labels)
    axs[2, 0].set_xlabel('time [s]')
    axs[2, 0].set_ylabel('ee_x')
    axs[2, 0].grid()

    axs[2, 1].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.vel_desired)[:-1, 1], color='red',
                linestyle='--', label='desired')
    axs[2, 1].plot(t_Nu[:-1], np.array(joint_velocity_ctrl.vel_actual)[1:, 1], color='blue',
                linestyle='--', label='actual')
    handles, labels = axs[2, 1].get_legend_handles_labels()
    axs[2, 1].legend(handles, labels)
    axs[2, 1].set_xlabel('time [s]')
    axs[2, 1].set_ylabel('ee_y')
    axs[2, 1].grid()

for i in range(dyn.Nx):
    axs[3, i].plot(t_Nu, np.array(joint_velocity_ctrl.x_dot_desired)[:, i], color='red',
                   linestyle='--', label='compute')
    handles, labels = axs[3, i].get_legend_handles_labels()
    axs[3, i].legend(handles, labels)
    axs[3, i].set_xlabel('time [s]')
    axs[3, i].set_ylabel('x%d' % i)
    axs[3, i].grid()

data_log = {
    'X_plot': X_plot,
    'U_plot': U_plot,
    'X_nom_val': X_nom_val,
    'X_future': X_future,
    'U_future': U_future
}

pickle.dump(data_log, open('./data/tracking_no_face_switch_data.npy', 'wb'))

plt.show()
#  -------------------------------------------------------------------

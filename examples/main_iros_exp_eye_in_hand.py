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
from std_srvs.srv import Trigger
from aruco_ros.srv import AskForMarkerCenter
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------
from geometry_msgs.msg import Twist
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
x_axis_in_world = np.array([np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
y_axis_in_world = np.array([-np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
z_axis_in_world = np.array([0., 0., -1.])
DEFAULT_ROTATION_MATRIX = np.c_[x_axis_in_world, \
                                y_axis_in_world, \
                                z_axis_in_world]
GOD_VIEW_TRANSLATION = np.array([0.50686447, -0.04996442, 0.53985807])
#  -------------------------------------------------------------------

# Panda ROS control
#  -------------------------------------------------------------------
def check_slider_marker_in_view():
    res = ask_for_marker1_in_view()
    return res.success

def acquire_slider_marker_center_in_image():
    res = ask_for_marker1_center()
    return np.array([res.marker_pose.x, res.marker_pose.y])

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
    fa.goto_pose(trans_step1, duration=3, ignore_virtual_walls=True)

    xyz_goal = trans
    xyz_goal_pre = trans_pre

    if trans_pre is not None:
        xyz_step2 = xyz_goal_pre + np.array([0., 0., 0.1])
        rotmat_step2 = DEFAULT_ROTATION_MATRIX
        trans_step2 = make_rigid_transform(xyz_step2, rotmat_step2, tool_frame_name)
        fa.goto_pose(trans_step2, duration=5, ignore_virtual_walls=True)
    else:
        xyz_step2 = xyz_goal + np.array([0., 0., 0.1])
        rotmat_step2 = DEFAULT_ROTATION_MATRIX
        trans_step2 = make_rigid_transform(xyz_step2, rotmat_step2, tool_frame_name)
        fa.goto_pose(trans_step2, duration=5, ignore_virtual_walls=True)

    xyz_step3 = xyz_goal.copy()
    rotmat_step3 = DEFAULT_ROTATION_MATRIX
    trans_step3 = make_rigid_transform(xyz_step3, rotmat_step3, tool_frame_name)
    fa.goto_pose(trans_step3, duration=3, ignore_virtual_walls=True)

    rospy.loginfo('Finished moving the pusher to new goal!')

    return trans_step3

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
            pre_rel_x = rel_x - 0.01
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'front':
        rel_x = 0.5*xl + rl
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x + 0.01
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'left':
        rel_y = 0.5*yl + rl
        rel_x = -rel_y * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x
            pre_rel_y = rel_y + 0.01
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'right':
        rel_y = -0.5*yl - rl
        rel_x = -rel_y * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x
            pre_rel_y = rel_y - 0.01
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    else:
        raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))

    if return_pre_pos:
        return rel_coords, pre_rel_coords
    else:
        return rel_coords

def get_abs_coords_on_slider(rel_coords, slider_pose):
    theta = slider_pose[2]
    xy_pos = slider_pose[:2]
    rot_mat = rotation_matrix(theta)
    abs_coords = xy_pos + rot_mat @ rel_coords

    return abs_coords

def get_psic(fa, tf):
    franka_pos = fa.get_pose()
    ## add bias to franka ee pose
    franka_pos.translation[0] -= 0.0
    franka_pos.translation[1] -= 0.0
    slider_pos, slider_ori = tf.get_slider_position_and_orientation()
    vec = rotation_matrix(slider_ori).T @ (franka_pos.translation - slider_pos)[0:2] + [beta[2], 0]
    phi = np.arctan2(-vec[1], -vec[0])
    return phi

# v_pub = rospy.Publisher("todo", Twist, queue_size=100)
def get_v_by_u0(u0, tf, fa:FrankaArm):
    phi = get_psic(fa, tf)
    w = np.array([u0[0], u0[1], u0[2]-u0[3]]).reshape(3, 1)
    xC, yC = get_rel_coords_on_slider(phi, beta=beta, contact_face="back", return_pre_pos=False)
    print("-------------------------------")
    print(f"xC: {xC}\nyC: {yC}\nphic: {phi}\nw: {w}")
    JC = np.array([1, 0, -yC,
                   0, 1, xC]).reshape(2, 3)
    GC = np.zeros((2, 3))
    # limit surface
    L = L_surf.copy()
    GC[:, 0:2] = (JC @ L @ JC.T)
    GC[:, 2] = np.array([0, xC/(np.cos(phi)*np.cos(phi))]).reshape(2)
    print('Gc: ', GC)
    v_S = np.zeros((3, 1))
    v_S[0:2] = GC @ w

    # scale the velocity v_x, v_y in slider coordinates
    v_S[0] = v_S[0] * 1.0
    # v_S[1] = max(-0.02, min(v_S[1], 0.02))
    v_S[1] = v_S[1] * (1/1.0)

    trans = tf.get_transform(tf.slider_frame_name, tf.base_frame_name)
    slider_quat = np.array([trans.transform.rotation.x, \
                            trans.transform.rotation.y, \
                            trans.transform.rotation.z, \
                            trans.transform.rotation.w])
    slider_rotmat = R.from_quat(slider_quat).as_matrix()
    v_G = slider_rotmat @ v_S / VELOCITY_SCALE  # (3,)
    print('v_S: ', v_S)
    print('v_G: ', v_G)
    # ---- compute q ----
    Ja = fa.get_jacobian(fa.get_joints())
    Ja_inv = np.linalg.pinv(Ja)

    # set predefined rate
    q_G = Ja_inv @ np.append([v_G], [0., 0., 0.])

    # q_G = Ja_inv @ np.array([0.02, 0., 0., 0., 0., 0.])
    return q_G, v_G

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
                trans = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('The transform from {0} to {1} does not exist!'.format(source_frame, target_frame))
                rospy.logwarn('The transform from {0} to {1} does not exist!'.format(source_frame, target_frame))
                rate.sleep()
                continue
        return trans

    def get_slider_position_and_orientation(self):
        """
        :return: slider_pos (x, y, z)
        :return: slider_ori theta
        """
        trans = self.get_transform(self.base_frame_name, self.slider_frame_name)
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

# ---- For Joint Velocity Control, usage similar to FrankaPyInterface
class JointVelocityControlInterface(object):
    def __init__(self, fa:FrankaArm, tf:tfHandler):
        self.fa = fa
        self.tf = tf
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.init_time = 0
        self.i = 0
        self.record = []
        self.vel_desired = []
        self.vel_actual = []

        self.v_G_lpf_coeff = 0.05
        self.last_v_G = None

        self.controller_type = None  # 'joint_velocity' or 'ee_pose'

    def refind_slider_out_of_view(self, steps=8):
        """
        Refind the slider marker in view
        """
        if check_slider_marker_in_view():
            print('Info: slider already in view!')
            return
        print('Info: cannot find the slider, trying to refind!')
        slider_maker_y_coords = []
        q7_list = []
        for q7 in np.linspace(FC.JOINT_LIMITS_MIN[-1]+0.05, FC.JOINT_LIMITS_MAX[-1]-0.05, steps):
            q7_list.append(q7)
            fa.goto_joints(np.append(fa.get_joints()[0:6], q7), duration=1, ignore_virtual_walls=True)
            if check_slider_marker_in_view():
                slider_marker_xy_in_image = acquire_slider_marker_center_in_image()
                slider_maker_y_coords.append(slider_marker_xy_in_image[1])
            else:
                slider_maker_y_coords.append(np.inf)
        if np.array(slider_maker_y_coords).min() >= 1e5:
            print('Warning: cannot refind the slider in camera!')
        q7_goal = q7_list[np.argmin(slider_maker_y_coords)]
        fa.goto_joints(np.append(fa.get_joints()[0:6], q7_goal), duration=3, ignore_virtual_walls=True)
        print('Info: refind the slider!')

    def panda_goto_god_view(self):
        """
        Control panda to god view
        """
        q7_home = FC.HOME_JOINTS[-1]
        self.fa.goto_joints(np.append(fa.get_joints()[0:6], q7_home), duration=1, ignore_virtual_walls=True)
        pose_cur = self.fa.get_pose()
        god_view_pose = RigidTransform(translation=GOD_VIEW_TRANSLATION, rotation=DEFAULT_ROTATION_MATRIX, \
                                        from_frame=pose_cur.from_frame, to_frame=pose_cur.to_frame)
        fa.goto_pose(god_view_pose, ignore_virtual_walls=True)

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
        self.panda_goto_god_view()
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

        ee_pose_cur = self.fa.get_pose()
        
        ee_pose_via_point = ee_pose_cur.copy()
        ee_pose_via_point.translation[0:2] = ee_abs_xy
        ee_pose_via_point.rotation = DEFAULT_ROTATION_MATRIX
        fa.goto_pose(ee_pose_via_point, ignore_virtual_walls=True)

        ee_pose_goal_point = ee_pose_via_point.copy()
        ee_pose_goal_point.translation[2] = slider_height - contact_point_depth
        ee_pose_goal_point.rotation = DEFAULT_ROTATION_MATRIX
        fa.goto_pose(ee_pose_goal_point, ignore_virtual_walls=True)
        print('Info: franka arrived at slider coarsely!')

    def goto_face_fine(self, contact_face):
        """
        Go to contact, based on finer pose measurement
        """
        self.refind_slider_out_of_view()
        slider_state, slider_height = self.get_slider_transform_fine()
        ee_rel_pos = self.get_ee_pos_in_contact(contact_face, get_pre_contact_pos=False)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = fa.get_pose()

        ee_pose_goal_point = ee_pose_cur.copy()
        ee_pose_goal_point.translation[0:2] = ee_abs_xy
        ee_pose_goal_point.translation[2] = slider_height - contact_point_depth
        ee_pose_goal_point.rotation = DEFAULT_ROTATION_MATRIX

        fa.goto_pose(ee_pose_goal_point, use_impedance=False, ignore_virtual_walls=True)
        self.refind_slider_out_of_view(steps=12)
        print('Info: franka arrived at slider finely!')

    def change_contact_face(self, contact_face):
        """
        Change another contact face
        """
        print('Info: change contact face!')
        self.goto_face_coarse(contact_face)
        self.goto_face_fine(contact_face)
    
    def joint_control_start(self):
        """
        Start joint velocity control skill
        """
        if self.controller_type is not None:
            raise ValueError('Controller already initialized!')
        self.controller_type = 'joint_velocity'
        self.home_joints = self.fa.get_joints()
        max_execution_time = 100
        self.fa.dynamic_joint_velocity(joints=self.home_joints,
                              joints_vel=np.zeros((7,)),
                              duration=max_execution_time,
                              buffer_time=10,
                              block=False)
        self.init_time = rospy.Time.now().to_time()
        self.i = 0

    def joint_control_stop_move(self):
        if self.controller_type != 'joint_velocity':
            raise ValueError('Controller type should be joint_velocity, but {0} called!'.format(self.controller_type))
        """
        Send zero velocity command
        Panda will stop move immediately
        """
        timestamp = rospy.Time.now().to_time() - self.init_time

        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=self.i, timestamp=timestamp,
            seg_run_time=30.0,
            joints=self.home_joints,
            joint_vels=np.zeros(7)
        )

        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        self.pub.publish(ros_msg)
        self.i += 1
        print('Info: panda stops move!')

    def pose_control_stop_move(self):
        if self.controller_type != 'ee_pose':
            raise ValueError('Controller type should be ee_pose, but {0} called!'.format(self.controller_type))
        print('Info: panda stops move!')
    
    def joint_control_go(self, u0, tf):
        """
        Calculate the desired velocity by limit surface F-V model
        """
        if self.controller_type != 'joint_velocity':
            raise ValueError('Controller type should be joint_velocity, but {0} called!'.format(self.controller_type))
        q_G, v_G = get_v_by_u0(u0, tf, self.fa)

        if self.last_v_G is None:
            v_G = v_G
        else:
            v_G = self.v_G_lpf_coeff * v_G + (1-self.v_G_lpf_coeff) * self.last_v_G
        self.last_v_G = v_G.copy()

        self.record.append(v_G)
        timestamp = rospy.Time.now().to_time() - self.init_time

        Jac = fa.get_jacobian(fa.get_joints())
        vel_desired = Jac @ q_G
        if np.linalg.norm(vel_desired) >= MAX_VELOCITY_DESIRED:
            q_G_scale_factor = np.linalg.norm(vel_desired) / MAX_VELOCITY_DESIRED
            q_G = q_G / q_G_scale_factor
        vel_desired = fa.get_jacobian(fa.get_joints()) @ q_G
        if np.linalg.norm(vel_desired) > MAX_VELOCITY_DESIRED + 1e-4:
            print('Warning: desired velocity exceed limits!')
            q_G = np.zeros_like(q_G)

        # record velocity value
        self.vel_desired.append((Jac @ q_G).reshape(-1,).tolist())
        self.vel_actual.append((Jac @ self.fa.get_joint_velocities()).reshape(-1,).tolist())

        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=self.i, timestamp=timestamp,
            seg_run_time=30.0,
            joints=self.home_joints,
            # joint_vels=np.zeros(7),
            joint_vels=q_G
        )

        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        self.pub.publish(ros_msg)
        self.i += 1

    def joint_control_go_delta_x(self, x0, x1, dt, tf):
        """
        Calculate the desired end effector velocity by (x1 - x0) / dt
        """
        # xy_ee = self.fa.get_pose().translation[:2]
        if self.controller_type != 'joint_velocity':
            raise ValueError('Controller type should be joint_velocity, but {0} called!'.format(self.controller_type))
        xy_ee = get_abs_coords_on_slider(
                        get_rel_coords_on_slider(x0[3], beta, contact_face='back', return_pre_pos=False),
                        x0[:3]
                    )
        xy_ee_next = get_abs_coords_on_slider(
                        get_rel_coords_on_slider(x1[3], beta, contact_face='back', return_pre_pos=False),
                        x1[:3]
                    )
        v_G = np.append((xy_ee_next - xy_ee) / dt, [0., 0., 0., 0.])

        # restrict velocity
        trans = tf.get_transform(tf.slider_frame_name, tf.base_frame_name)
        slider_quat = np.array([trans.transform.rotation.x, \
                                trans.transform.rotation.y, \
                                trans.transform.rotation.z, \
                                trans.transform.rotation.w])
        slider_rotmat = R.from_quat(slider_quat).as_matrix()
        vxyz_S = slider_rotmat.T @ v_G[0:3]
        vxyz_S[1] = max(-0.02, min(vxyz_S[1], 0.02))
        v_G[0:3] = slider_rotmat @ vxyz_S

        # if self.last_v_G is None:
        #     v_G = v_G
        # else:
        #     v_G = self.v_G_lpf_coeff * v_G + (1-self.v_G_lpf_coeff) * self.last_v_G
        # self.last_v_G = v_G.copy()

        self.record.append(v_G)
        timestamp = rospy.Time.now().to_time() - self.init_time

        Jac = fa.get_jacobian(fa.get_joints())
        q_G = np.linalg.pinv(Jac) @ v_G

        vel_desired = Jac @ q_G
        if np.linalg.norm(vel_desired) >= MAX_VELOCITY_DESIRED:
            q_G_scale_factor = np.linalg.norm(vel_desired) / MAX_VELOCITY_DESIRED
            q_G = q_G / q_G_scale_factor
        vel_desired = fa.get_jacobian(fa.get_joints()) @ q_G
        if np.linalg.norm(vel_desired) > MAX_VELOCITY_DESIRED + 1e-4:
            print('Warning: desired velocity exceed limits!')
            q_G = np.zeros_like(q_G)

        # record velocity value
        self.vel_desired.append((Jac @ q_G).reshape(-1,).tolist())
        self.vel_actual.append((Jac @ self.fa.get_joint_velocities()).reshape(-1,).tolist())

        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=self.i, timestamp=timestamp,
            seg_run_time=30.0,
            joints=self.home_joints,
            # joint_vels=np.zeros(7),
            joint_vels=q_G
        )

        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        self.pub.publish(ros_msg)
        self.i += 1
    
    def joint_control_terminate(self):
        """
        Send termination message and kill the skill.
        If you do not kill the skill, an error will be raised for skill conflicts
        """
        if self.controller_type != 'joint_velocity':
            raise ValueError('Controller type should be joint_velocity, but {0} called!'.format(self.controller_type))
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)
        self.fa.stop_skill()

    def pose_control_start(self):
        if self.controller_type is not None:
            raise ValueError('Controller already initialized!')
        self.controller_type = 'ee_pose'
        self.home_pose = self.fa.get_pose()
        max_execution_time = 100
        self.fa.goto_pose(tool_pose=self.home_pose,
                          duration=max_execution_time,
                          use_impedance=False,
                          dynamic=True,
                          buffer_time=10,
                          block=False,
                          ignore_virtual_walls=True)
        self.pusher_tip_height = self.home_pose.translation[2]  # z-coordinate of the pusher
        self.init_time = rospy.Time.now().to_time()
        self.i = 0

    def pose_control_go(self, x1):
        if self.controller_type != 'ee_pose':
            raise ValueError('Controller type should be ee_pose, but {0} called!'.format(self.controller_type))
        
        Jac = fa.get_jacobian(fa.get_joints())
        q_G = fa.get_joint_velocities()
        v_G = Jac @ q_G
        self.record.append(v_G)

        xy_ee_next = get_abs_coords_on_slider(
                        get_rel_coords_on_slider(x1[3], beta, contact_face='back', return_pre_pos=False),
                        x1[:3]
                    )

        pose_cur = self.fa.get_pose()
        xy_ee = pose_cur.translation[0:2]

        xy_ee_next = xy_ee_next + 0.01 * (xy_ee_next - xy_ee)

        pose = RigidTransform(translation=np.append(xy_ee_next, self.pusher_tip_height), rotation=DEFAULT_ROTATION_MATRIX, \
                              from_frame=pose_cur.from_frame, to_frame=pose_cur.to_frame)

        if np.linalg.norm(pose_cur.translation[0:2]-xy_ee_next) > MAX_TRANSLATION_DESIRED:
            print('Warning: maximum translation limit exceeded, panda will stop move!')
            pose = pose_cur.copy()

        timestamp = rospy.Time.now().to_time() - self.init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=self.i, timestamp=timestamp, 
            position=pose.translation, quaternion=pose.quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=self.i, timestamp=timestamp,
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
        self.i += 1

    def pose_control_terminite(self):
        if self.controller_type != 'ee_pose':
            raise ValueError('Controller type should be ee_pose, but {0} called!'.format(self.controller_type))
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)
    

class FrankaPyInterface(object):
    def __init__(self, fa:FrankaArm) -> None:
        self.fa = fa
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    
    def pose_control_start(self, pose0):
        fa.goto_pose(pose0, duration=100, dynamic=True, buffer_time=10,
            cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
            ignore_virtual_walls=True
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
T = 35  # time of the simulation is seconds
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
contact_point_depth = 0.015
L_SURF_SCALE = 1.3888888888888888
# L_SURF_SCALE = 1.
VELOCITY_SCALE = 1.
MAX_VELOCITY_DESIRED = 0.05 / 3.5
MAX_TRANSLATION_DESIRED = 0.1  # the maximum translation when running (dynamic) goto_pose

LARGER_TRANSLATIONAL_STIFFNESS = [600.0, 600.0, 600.0]
LARGER_ROTATIONAL_STIFFNESS = [50.0, 50.0, 50.0]

## disturbance observer gain
K_d = 0.0  # set K_d>0 to enable DOB

# controller_type = 'ee_pose'
controller_type = 'ee_pose'

base_frame_name = 'panda_link0'
slider_frame_name = 'marker1_frame'
#  -------------------------------------------------------------------

# initialize TF2_ROS interface
#  -------------------------------------------------------------------
fa = FrankaArm(with_gripper=False, ros_log_level=rospy.DEBUG)
tf_handler = tfHandler(base_frame_name=base_frame_name,
                       slider_frame_name=slider_frame_name)
#  -------------------------------------------------------------------

# initialize Franka control
#  -------------------------------------------------------------------
franka_ctrl = FrankaPyInterface(fa)
joint_velocity_ctrl = JointVelocityControlInterface(fa, tf_handler)
rospy.wait_for_service('/aruco_simple/ask_for_pose_ready')
rospy.wait_for_service('/aruco_simple/ask_for_marker_center')
ask_for_marker1_in_view = rospy.ServiceProxy('/aruco_simple/ask_for_pose_ready', Trigger)
ask_for_marker1_center = rospy.ServiceProxy('/aruco_simple/ask_for_marker_center', AskForMarkerCenter)
#  -------------------------------------------------------------------

# initialize Franka pose
#  -------------------------------------------------------------------
joint_velocity_ctrl.change_contact_face('back')
rospy.loginfo('Move the pusher to start position!')
if controller_type == 'joint_velocity':
    joint_velocity_ctrl.joint_control_start()
elif controller_type == 'ee_pose':
    joint_velocity_ctrl.pose_control_start()
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
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.35, 0.0, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_sine(0.35, 0.0, 0.04, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_ellipse(-np.pi/2, 3*np.pi/2, 0.2, 0.1, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.3, N, N_MPC)
x0_nom += x_init_val[0]
x1_nom += x_init_val[1]
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# Compute nominal actions for sticking contact
#  ------------------------------------------------------------------
dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        planning_config['dynamics'],
        planning_config['TO']['contactMode'],
        pusherAngleLim=tracking_config['dynamics']['xFacePsiLimit']
)
optObjNom = sliding_pack.to.buildOptObj(
        dynNom, N+N_MPC, planning_config['TO'], dt=dt, max_iter=180)

# initialize limit surface matrix
L_surf = dyn.A(beta).toarray() * L_SURF_SCALE

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
        X_nom_val, None, dt=dt, max_iter=40
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
comp_time = np.empty((Nidx-1, 1))
success = np.empty(Nidx-1)
cost_plot = np.empty((Nidx-1, 1))
#  -------------------------------------------------------------------

# Disturbance
#  -------------------------------------------------------------------
D_hat_plot = np.empty([dyn.Nx, Nidx])  # observed disturbance
D_true_plot = np.empty([dyn.Nx, Nidx])  # actual disturbance
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
import pdb
pdb.set_trace()
rosrate = rospy.Rate(freq)
x0 = x_init_val
# last_state = x0.copy()

## initialize disturbance
d0 = [0.] * dyn.Nx
D_hat_plot[:, 0] = d0
D_true_plot[:, 0] = [0.]*dyn.Nx
x1_estimation = None

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

    slider_abs_pos, slider_abs_ori = tf_handler.get_slider_position_and_orientation()
    psic = get_psic(fa, tf_handler)
    x0 = [slider_abs_pos[0], slider_abs_pos[1], slider_abs_ori, psic]
    
    ## update disturbance estimation
    if x1_estimation is not None:
        d_hat = (np.array(d_hat)+K_d*(np.array(x0) - np.array(x1_estimation)) / dt).tolist()
    D_hat_plot[:, idx+1] = d_hat
    D_true_plot[:, idx+1] = np.zeros(dyn.Nx)

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
    x1_estimation = x_opt[:, 1].elements()

    # ---- send joint velocity by u0 (v0) ----
    if controller_type == 'joint_velocity':
        joint_velocity_ctrl.joint_control_go(u0, tf_handler)
    elif controller_type == 'ee_pose':
        x1 = (x0 + (dyn.f(x0, u0, beta) + d0)*dt).elements()
        joint_velocity_ctrl.pose_control_go(x1)
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
#  -------------------------------------------------------------------
# stop joint velocity control interface
if controller_type == 'joint_velocity':
    joint_velocity_ctrl.joint_control_terminate()
elif controller_type == 'ee_pose':
    joint_velocity_ctrl.pose_control_terminite()
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
    ani.save('./videos/MPC_MPCC_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
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
# #  -------------------------------------------------------------------
# # plot position
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
# # plot actions
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
fig, axs = plt.subplots(3, 4, sharex=True)

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

plt.show()

np.save('./data/x_traj.npy', X_plot)
np.save('./data/u_traj.npy', U_plot)
np.save('./data/X_nom_val.npy', X_nom_val)
#  -------------------------------------------------------------------

import numpy as np
from scipy.spatial.transform import Rotation as R

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from sliding_pack.utils.utils import *

#  -------------------------------------------------------------------
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
#  -------------------------------------------------------------------

sys.path = ['/home/roboticslab/jyp/catkin_ws/devel/lib/python3/dist-packages'] + sys.path
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

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

# Initialize constants
#  -------------------------------------------------------------------
contact_point_depth = 0.02
L_SURF_SCALE = 1.3888888888888888
# L_SURF_SCALE = 1.
VELOCITY_SCALE = 10.
MAX_VELOCITY_DESIRED = 0.5

ur_robot_ip = '192.168.101.50'
base_frame_name = 'world'
ur5_base_frame_name = 'base'
slider_frame_name = 'marker1_frame'
default_contact_face = 'back'

controller_type = 'speedL'
#  -------------------------------------------------------------------

def rvecpose2rtrans(pose):
    pose_rtrans = RigidTransform()
    pose_rtrans.translation = pose[0:3]
    pose_rtrans.rotation = R.from_rotvec(pose[3:6]).as_matrix()
    return pose_rtrans

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

    def rotate_wrist3_to_find_slider(self, slider_theta, contact_face):
        if contact_face == 'front':
            slider_theta += np.pi
        elif contact_face == 'back':
            slider_theta += 0
        elif contact_face == 'left':
            slider_theta += -np.pi/2
        elif contact_face == 'right':
            slider_theta += np.pi/2
        else:
            raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))

        slider_theta = restrict_angle_in_unit_circle(slider_theta)
        assert -np.pi <= slider_theta <= np.pi

        if 2.07 <= slider_theta <= np.pi:
            wrist3_angle = -slider_theta + 5 * np.pi / 4
        elif -np.pi <= slider_theta <= 1.05:
            wrist3_angle = -slider_theta - 3 * np.pi / 4
        elif 1.5 <= slider_theta <= 2.07:
            wrist3_angle = 1.85
        elif 1.05 <= slider_theta <= 1.5:
            wrist3_angle = -3.4
        else:
            raise ValueError('The slider theta {0} should be in [-pi, pi]!'.format(slider_theta))
        Q_ur5_now = self.receive.getActualQ()
        Q_ur5_now[-1] = wrist3_angle

        assert -3.4 <= Q_ur5_now[-1] <= 1.85

        success = self.control.moveJ(Q_ur5_now)
        if not success:
            rospy.logwarn('Command moveJ failed!')

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

    def stop_current_move(self):
        self.control.speedStop(10.0)
        rospy.loginfo('UR5 has stopped successfully!')

class UR5ControlInterface(object):
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

    def _set_L_surf(self, L_surf):
        self.L_surf = L_surf

    def _get_psic(self, contact_face):
        # in world frame
        ur5_pos = self.ur.get_pose()
        slider_pos, slider_theta = self.tf.get_slider_position_and_orientation()
        # in slider frame
        bias_pusher_contact_point = self._get_contact_point_bias(contact_face)
        contact_point = rotation_matrix(slider_theta).T @ (ur5_pos.translation - slider_pos)[0:2] + bias_pusher_contact_point
        pusher_phic = self._convert_contact_point_to_psic(contact_point, contact_face)
        return pusher_phic

    def get_contact_jacobian(self, psic, contact_face):
        xc, yc = get_contact_point_on_slider(psic, self.beta, contact_face)
        Jc = np.array([[1, 0, -yc],
                       [0, 1, xc]])
        return Jc

    """
    @ comment: the bias from pusher tip to actual contact point, in slider frame
    """
    def _get_contact_point_bias(self, contact_face):
        if contact_face == 'front':
            _bias = [-self.beta[2], 0]
        elif contact_face == 'back':
            _bias = [self.beta[2], 0]
        elif contact_face == 'left':
            _bias = [0, -self.beta[2]]
        elif contact_face == 'right':
            _bias = [0, self.beta[2]]
        else:
            raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))
        return _bias

    """
    @ comment: convert contact point to psic, in slider frame
    """
    def _convert_contact_point_to_psic(self, contact_point, contact_face):
        px, py = contact_point
        if contact_face == 'front':
            psic = np.arctan2(py, px)
        elif contact_face == 'back':
            psic = np.arctan2(-py, -px)
        elif contact_face == 'left':
            psic = np.arctan2(-px, py)
        elif contact_face == 'right':
            psic = np.arctan2(px, -py)
        else:
            raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))
        return psic

    """
    @ comment: return actual theta and psic \in [-psic_lim, +psic_lim]
    """
    def _get_slider_and_pusher_state(self, contact_face):
        # in world frame
        ur5_pos = self.ur.get_pose()
        slider_pos, slider_theta = self.tf.get_slider_position_and_orientation()
        # in slider frame
        bias_pusher_contact_point = self._get_contact_point_bias(contact_face)
        contact_point = rotation_matrix(slider_theta).T @ (ur5_pos.translation - slider_pos)[0:2] + bias_pusher_contact_point
        pusher_phic = self._convert_contact_point_to_psic(contact_point, contact_face)
        return [slider_pos[0], slider_pos[1], slider_theta, pusher_phic]

    def _ur5_goto_god_view(self, god_view_just_lift_up=False):
        """
        Control ur5 to god view
        """
        # lift up to detach the slider
        pose_cur = self.ur.get_pose()
        
        if god_view_just_lift_up:
            pose_liftup = pose_cur.copy()
            if pose_liftup.translation[2] < -0.55:
                pose_liftup.translation[2] += 0.1
                self.ur.goto_pose(pose_liftup)
        else:
            pose_liftup = pose_cur.copy()
            if pose_liftup.translation[2] < -0.55:
                pose_liftup.translation[2] += 2 * contact_point_depth
                self.ur.goto_pose(pose_liftup)

            pos_xyz_rvec_home = GOD_VIEW_XYZ_RVEC
            pose_rtrans = rvecpose2rtrans(pos_xyz_rvec_home)
            self.ur.goto_pose(pose_rtrans, use_world_frame=False)

    def _get_ee_pos_in_contact(self, contact_face, get_pre_contact_pos=True):
        """
        Get the ee position when in contact
        """
        contact_ee_rel_pos, pre_contact_ee_rel_pos = get_rel_coords_on_slider(0., self.beta, contact_face, return_pre_pos=True)
        if get_pre_contact_pos:
            return pre_contact_ee_rel_pos
        else:
            return contact_ee_rel_pos

    def _get_slider_transform_coarse(self, god_view_just_lift_up=False):
        """
        Get coarse slider pose
        """
        self._ur5_goto_god_view(god_view_just_lift_up)
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def _get_slider_transform_fine(self):
        """
        Get finer slider pose
        """
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def _goto_face_coarse(self, contact_face, god_view_just_lift_up=False):
        """
        Go to contact, based on coarse pose measurement
        """
        slider_state, slider_height = self._get_slider_transform_coarse(god_view_just_lift_up)
        ee_rel_pos = self._get_ee_pos_in_contact(contact_face, get_pre_contact_pos=True)
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

        self.ur.rotate_wrist3_to_find_slider(slider_state[2], contact_face)
        print('Info: franka arrived at slider coarsely!')

    def _goto_face_fine(self, contact_face):
        """
        Go to contact, based on finer pose measurement
        """
        slider_state, slider_height = self._get_slider_transform_fine()
        ee_rel_pos = self._get_ee_pos_in_contact(contact_face, get_pre_contact_pos=False)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = self.ur.get_pose()

        ee_pose_goal_point = ee_pose_cur.copy()
        ee_pose_goal_point.translation[0:2] = ee_abs_xy
        ee_pose_goal_point.translation[2] = slider_height - contact_point_depth
        ee_pose_goal_point.rotation = ee_pose_cur.rotation

        self.ur.goto_pose(ee_pose_goal_point)
        print('Info: franka arrived at slider finely!')

    def _change_contact_face(self, contact_face, god_view_just_lift_up=False):
        """
        Change another contact face
        """
        print('Info: change contact face!')
        self._goto_face_coarse(contact_face, god_view_just_lift_up)
        self._goto_face_fine(contact_face)
    
    def _get_input_in_slider_frame(self, u, contact_face):
        if contact_face == 'front':
            u_ext = [-u[0], -u[1], u[2]-u[3]]
        elif contact_face == 'back':
            u_ext = [u[0], u[1], u[2]-u[3]]
        elif contact_face == 'left':
            u_ext = [u[1], -u[0], u[2]-u[3]]
        elif contact_face == 'right':
            u_ext = [-u[1], u[0], u[2]-u[3]]
        else:
            raise NotImplementedError('The contact face {0} is not supported!'.format(contact_face))
        return u_ext

    def _velocity_control_go(self, u, contact_face):
        psic = self._get_psic(contact_face)
        u_extend = self._get_input_in_slider_frame(u, contact_face)
        Jc = self.get_contact_jacobian(psic, contact_face)
        Gc = np.zeros((2, 3))
        Gc[:, 0:2] = (Jc @ self.L_surf @ Jc.T)
        xc = Jc[1, 2]; yc = -Jc[0, 2]

        if contact_face == 'back' or contact_face == 'front':
            Gc[:, 2] = np.array([0, xc/(np.cos(psic)*np.cos(psic))])
        elif contact_face == 'left' or contact_face == 'right':
            Gc[:, 2] = np.array([-yc/(np.cos(psic)*np.cos(psic)), 0])

        v_desired = Gc @ u_extend  # in slider frame
        _, slider_theta = self.tf.get_slider_position_and_orientation()
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

    def _velocity_control_stop(self):
        # success = self.ur.apply_velocity_ee([0., 0.])
        # if not success:
        #     rospy.logwarn('Failed to control UR5 with velocity {0}!'.format([0., 0.]))
        self.ur.stop_current_move()

    # tag: external function
    def ur5_move_to_contact_face(self, contact_face, god_view_just_lift_up=False):
        self._change_contact_face(contact_face, god_view_just_lift_up)

    # tag: external function
    def ur5_get_state_variable(self, contact_face):
        return self._get_slider_and_pusher_state(contact_face)

    # tag: external function
    def ur5_apply_velocity_control(self, u, contact_face):
        self._velocity_control_go(u, contact_face)

    # tag: external function
    def ur5_stop_move(self):
        self._velocity_control_stop()

def initialize_tf_and_control(beta, dt) -> UR5ControlInterface:
    # initialize TF2_ROS interface
    #  -------------------------------------------------------------------
    nh = rospy.init_node('pusher_slider_controller', anonymous=True)
    
    ur = UR5Arm(robot_ip=ur_robot_ip)
    ur.set_base_in_world(T_URBASE2WORLD)
    tf_handler = tfHandler(base_frame_name=base_frame_name,
                           slider_frame_name=slider_frame_name,
                           ur=ur)
    
    ur5_interface = UR5ControlInterface(ur=ur,
                                        tf=tf_handler,
                                        beta=beta,
                                        dt=dt)
    return ur5_interface

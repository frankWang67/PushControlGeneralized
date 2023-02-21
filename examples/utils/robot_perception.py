import numpy as np
from scipy.spatial.transform import Rotation as R

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from sliding_pack.utils.utils import *

sys.path = ['/home/roboticslab/jyp/catkin_ws/devel/lib/python3/dist-packages'] + sys.path
import rospy
import tf2_ros
from std_srvs.srv import Trigger

# Initialize constants
#  -------------------------------------------------------------------
x_axis_in_world = np.array([np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
y_axis_in_world = np.array([-np.sqrt(2)/2., -np.sqrt(2)/2., 0.])
z_axis_in_world = np.array([0., 0., -1.])
DEFAULT_ROTATION_MATRIX = np.c_[x_axis_in_world, \
                                y_axis_in_world, \
                                z_axis_in_world]
GOD_VIEW_TRANSLATION = np.array([0.50686447, -0.04996442, 0.53985807])

GOD_VIEW_JOINTS1 = np.array([0.00169938, -0.13103018, -0.08288011, -1.41510536, \
                             -0.01651296, 1.60939038,  0.71218718])
GOD_VIEW_JOINTS2 = np.array([0.00169938, -0.13103018, -0.08288011, -1.41510536, \
                             -0.01651296, 1.30939038,  0.71218718])

MAX_TRANSLATION_DESIRED = 0.1

PANDA_JOINT7_MIN = -1.2036
PANDA_JOINT7_MAX = 2.7426

base_frame_name = 'panda_link0'
slider_frame_name = 'marker1_frame'
aruco_pose_ready_server_name = '/aruco_simple/ask_for_pose_ready'
#  -------------------------------------------------------------------

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
    rl_used = 0.014
    if contact_face == 'back':
        rel_x = -0.5*xl - rl_used
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x - 0.02
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'front':
        rel_x = 0.5*xl + rl_used
        rel_y = rel_x * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x + 0.02
            pre_rel_y = rel_y
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'left':
        rel_y = 0.5*yl + rl_used
        rel_x = -rel_y * np.tan(psic)
        rel_coords = np.array([rel_x, rel_y])
        if return_pre_pos:
            pre_rel_x = rel_x
            pre_rel_y = rel_y + 0.02
            pre_rel_coords = np.array([pre_rel_x, pre_rel_y])
    elif contact_face == 'right':
        rel_y = -0.5*yl - rl_used
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
    def __init__(self, base_frame_name, slider_frame_name, aruco_pose_ready_server_name) -> None:
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.slider_frame_name = slider_frame_name
        self.base_frame_name = base_frame_name
        self.aruco_pose_ready_server_name = aruco_pose_ready_server_name

        rospy.wait_for_service('/aruco_simple/ask_for_marker_center')
        self.ask_for_marker1_in_view = rospy.ServiceProxy('/aruco_simple/ask_for_pose_ready', Trigger)

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

    # Panda ROS control
    #  -------------------------------------------------------------------
    def check_slider_marker_in_view(self):
        res = self.ask_for_marker1_in_view()
        return res.success

    

# ---- For Franka Panda Control
class PandaControlInterface(object):
    def __init__(self, fa:FrankaArm, tf:tfHandler, beta):
        self.fa = fa
        self.tf = tf

        self.beta = beta
        self.pusher_radius = self.beta[2]
        self.theta_offset = {'back': 0., 'front': np.pi, 'left': -np.pi/2, 'right': np.pi/2}

        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.contact_point_depth = 0.015
        self.init_time = 0
        self.i = 0

        self.panda_state = 'stopped'

    def _refind_slider_out_of_view(self, contact_face):
        """
        Refind the slider marker in view
        """
        import pdb; pdb.set_trace()
        rospy.loginfo('Trying to refind the slider in view!')

        # lift up the pusher
        rospy.loginfo('Lifting up the pusher!')
        pose_cur = self.fa.get_pose()
        pose_next = pose_cur.copy()
        pose_next.translation[2] += 0.1
        self.fa.goto_pose(pose_next, ignore_virtual_walls=True)

        # rotate the last joint
        rospy.loginfo('Rotate the last joint!')
        _, slider_theta = self._get_slider_transform_coarse()
        if slider_theta >= (np.pi/4-PANDA_JOINT7_MIN):
            panda_joint7 = PANDA_JOINT7_MIN
        elif slider_theta <= (np.pi/4-PANDA_JOINT7_MAX):
            panda_joint7 = PANDA_JOINT7_MAX
        else:
            panda_joint7 = np.pi/4 - slider_theta
        self.fa.goto_joints(np.append(self.fa.get_joints()[0:6], panda_joint7), duration=3, ignore_virtual_walls=True)

        # recontact
        rospy.loginfo('Re-contact with the slider!')
        self._goto_face_coarse(contact_face)
        self._goto_face_fine(contact_face)

    def _adjust_joint7_to_find_slider(self, slider_theta):
        slider_theta = restrict_angle_in_unit_circle(slider_theta)
        # if slider_theta >= (np.pi/4-PANDA_JOINT7_MIN):
        if (np.pi/4-PANDA_JOINT7_MIN) <= slider_theta <= np.pi or -np.pi <= slider_theta <= -3.1256:
            panda_joint7 = PANDA_JOINT7_MIN
        elif -3.1256 <= slider_theta <= (np.pi/4-PANDA_JOINT7_MAX):
            panda_joint7 = PANDA_JOINT7_MAX
        else:
            panda_joint7 = np.pi/4 - slider_theta
        self.fa.goto_joints(np.append(self.fa.get_joints()[0:6], panda_joint7), duration=3, ignore_virtual_walls=True)

    def _panda_goto_god_view(self):
        """
        Control panda to god view
        """
        pose_cur = self.fa.get_pose()
        pose_next = pose_cur.copy()
        pose_next.translation[2] += 0.1
        self.fa.goto_pose(pose_next, ignore_virtual_walls=True)

        q7_home = FC.HOME_JOINTS[-1]
        self.fa.goto_joints(np.append(self.fa.get_joints()[0:6], q7_home), duration=3, ignore_virtual_walls=True)
        pose_cur = self.fa.get_pose()
        god_view_pose = RigidTransform(translation=GOD_VIEW_TRANSLATION, rotation=DEFAULT_ROTATION_MATRIX, \
                                        from_frame=pose_cur.from_frame, to_frame=pose_cur.to_frame)
        self.fa.goto_pose(god_view_pose, ignore_virtual_walls=True)

        find_slider_flag = False
        _rate = rospy.Rate(30)
        # go to the first position
        self.fa.goto_joints(GOD_VIEW_JOINTS1, ignore_virtual_walls=True)
        # try to find the slider
        if not find_slider_flag:
            for _ in range(10):
                if self.tf.check_slider_marker_in_view():
                    rospy.loginfo('Find the slider!')
                    find_slider_flag = True
                    break
                _rate.sleep()
        
        # go to the second position
        if not find_slider_flag:
            self.fa.goto_joints(GOD_VIEW_JOINTS2, ignore_virtual_walls=True)
            for _ in range(10):
                if self.tf.check_slider_marker_in_view():
                    rospy.loginfo('Find the slider!')
                    break
                _rate.sleep()

        if not find_slider_flag:
            rospy.logerr('Cannot find the slider!')

    def _get_ee_pos_in_contact(self, contact_face, get_pre_contact_pos=True):
        """
        Get the ee position when in contact
        """
        contact_ee_rel_pos, pre_contact_ee_rel_pos = get_rel_coords_on_slider(0., self.beta, contact_face, return_pre_pos=True)
        if get_pre_contact_pos:
            return pre_contact_ee_rel_pos
        else:
            return contact_ee_rel_pos

    def _get_slider_transform_coarse(self):
        """
        Get coarse slider pose
        """
        self._panda_goto_god_view()
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def _get_slider_transform_fine(self):
        """
        Get finer slider pose
        """
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()
        return np.array([slider_pos[0], slider_pos[1], slider_ori]), slider_pos[2]

    def _goto_face_coarse(self, contact_face):
        """
        Go to contact, based on coarse pose measurement
        """
        slider_state, slider_height = self._get_slider_transform_coarse()
        ee_rel_pos = self._get_ee_pos_in_contact(contact_face, get_pre_contact_pos=True)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = self.fa.get_pose()
        
        ee_pose_via_point = ee_pose_cur.copy()
        ee_pose_via_point.translation[0:2] = ee_abs_xy
        ee_pose_via_point.rotation = DEFAULT_ROTATION_MATRIX
        self.fa.goto_pose(ee_pose_via_point, ignore_virtual_walls=True)

        ee_pose_goal_point = ee_pose_via_point.copy()
        ee_pose_goal_point.translation[2] = slider_height - self.contact_point_depth
        ee_pose_goal_point.rotation = DEFAULT_ROTATION_MATRIX
        self.fa.goto_pose(ee_pose_goal_point, ignore_virtual_walls=True)

        self._adjust_joint7_to_find_slider(slider_state[2]+self.theta_offset[contact_face])
        print('Info: franka arrived at slider coarsely!')

    def _goto_face_fine(self, contact_face):
        """
        Go to contact, based on finer pose measurement
        """
        slider_state, slider_height = self._get_slider_transform_fine()
        ee_rel_pos = self._get_ee_pos_in_contact(contact_face, get_pre_contact_pos=False)
        ee_abs_xy = get_desired_end_effector_xy_abs(slider_state, ee_rel_pos)

        ee_pose_cur = self.fa.get_pose()

        ee_pose_goal_point = ee_pose_cur.copy()
        ee_pose_goal_point.translation[0:2] = ee_abs_xy
        ee_pose_goal_poindef panda_get_state_variable(self, contact_face):
        return self._get_slider_and_pusher_state(contact_face)
    def _change_contact_face(self, contact_face):
        """
        Change another contact face
        """
        print('Info: change contact face!')
        self._goto_face_coarse(contact_face)
        self._goto_face_fine(contact_face)
    
    def _pose_control_start(self):
        self.home_pose = self.fa.get_pose()
        max_execution_time = 100

        self.fa.goto_pose(tool_pose=self.home_pose,
                          duration=max_execution_time,
                          use_impedance=False,
                          dynamic=True,
                          buffer_time=10,
                          block=False,
                          ignore_virtual_walls=True)
        self.panda_state = 'started'

        self.pusher_tip_height = self.home_pose.translation[2]  # z-coordinate of the pusher
        self.init_time = rospy.Time.now().to_time()
        self.i = 0

    def _pose_control_go(self, x1):
        Jac = self.fa.get_jacobian(self.fa.get_joints())
        q_G = self.fa.get_joint_velocities()
        v_G = Jac @ q_G

        xy_ee_next = get_abs_coords_on_slider(
                        get_rel_coords_on_slider(x1[3], self.beta, contact_face='back', return_pre_pos=False),
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

    def _pose_control_terminite(self):
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)
        self.panda_state = 'stopped'

    def _get_slider_and_pusher_state(self, contact_face):
        if contact_face is None:
            raise ValueError('Contact face is not provided!')

        franka_pos = self.fa.get_pose()
        slider_pos, slider_ori = self.tf.get_slider_position_and_orientation()

        # offset theta according to contact face
        slider_ori = slider_ori

        vec = rotation_matrix(slider_ori).T @ (franka_pos.translation - slider_pos)[0:2] + [self.pusher_radius, 0]
        pusher_phic = np.arctan2(-vec[1], -vec[0])
        return [slider_pos[0], slider_pos[1], slider_ori, pusher_phic]

    # ----- external API -----
    def panda_move_to_contact_face(self, contact_face):
        self._change_contact_face(contact_face)

    def panda_apply_control(self, x1):
        self._pose_control_go(x1)

    def panda_start_move(self):
        if self.panda_state == 'stopped':
            self._pose_control_start()
            print('Info: panda dynamic pose control started!')
        else:
            pass

    def panda_stop_move(self):
        if self.panda_state == 'started':
            self._pose_control_terminite()
            print('Info: panda dynamic pose control terminated!')
        else:
            pass

    def panda_get_state_variable(self, contact_face):
        return self._get_slider_and_pusher_state(contact_face)

    def panda_refind_slider_and_recontact(self, contact_face):
        rospy.sleep(1)
        self.panda_stop_move()
        self._refind_slider_out_of_view(contact_face)
        self.panda_start_move()

def initialize_tf_and_control(beta) -> PandaControlInterface:
    # initialize TF2_ROS interface
    #  -------------------------------------------------------------------
    fa = FrankaArm(with_gripper=False, ros_log_level=rospy.DEBUG, init_node=True)
    tf_handler = tfHandler(base_frame_name=base_frame_name,
                            slider_frame_name=slider_frame_name,
                            aruco_pose_ready_server_name=aruco_pose_ready_server_name)
    panda_interface = PandaControlInterface(fa=fa,
                                            tf=tf_handler,
                                            beta=beta)
    return panda_interface

"""
    Save the poses of slider and obstacles
    for real robot experiment
"""

import casadi as cs
import numpy as np
import sys
import pickle
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3

# ----- use sliding pack -----
from sliding_pack.utils.intfunc import *

# ----- use tf2 in python3 -----
sys.path = ['/home/roboticslab/jyp/catkin_ws/devel/lib/python3/dist-packages'] + sys.path
import rospy
import tf2_ros

# configurations
# ----------------------------------------------------
# base_frame = 'panda_link0'
base_frame = 'world'
slider_frame = 'marker1_frame'
movable_frame = 'marker2_frame'
immovable_frame1 = 'marker3_frame'
immovable_frame2 = 'marker4_frame'

num_obs = 3

geom_target = [0.08, 0.15]
geom_movable = [0.07, 0.122]
geom_immovable = [0.10, 0.102]

miu_movable = 0.3
miu_immovable = 0.3

dt_contact = 0.05

# ----------------------------------------------------

def position2array(position:Vector3):
    return np.array([position.x, position.y, position.z])

def quaternion2array(quat:Quaternion):
    return np.array([quat.x, quat.y, quat.z, quat.w])

def quaternion2theta(quat):
    return R.from_quat(quat).as_euler('xyz')[2]

def compute_limit_surface(geom):
    __geom = cs.SX.sym('geom', 2)
    __c = rect_cs(__geom[0], __geom[1])/(__geom[0]*__geom[1])
    __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
    __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2)
    A = cs.Function('A', [__geom], [__A])
    lim_surf_A_obs = A(geom).toarray()
    return lim_surf_A_obs

class tfHandler(object):
    def __init__(self) -> None:
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def get_transform(self, target_frame, source_frame) -> TransformStamped:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('The transform from {0} to {1} does not exist!'.format(source_frame, target_frame))
                rospy.logwarn_throttle(1.0, 'The transform from {0} to {1} does not exist!'.format(source_frame, target_frame))
                rate.sleep()
                continue
        return trans

nh = rospy.init_node('planning_scene_node', anonymous=True)
tf_handler = tfHandler()
trans_slider = tf_handler.get_transform(base_frame, slider_frame)
trans_movable = tf_handler.get_transform(base_frame, movable_frame)
trans_immovable1 = tf_handler.get_transform(base_frame, immovable_frame1)
trans_immovable2 = tf_handler.get_transform(base_frame, immovable_frame2)

# slider
quat0 = quaternion2array(trans_slider.transform.rotation)
theta0 = quaternion2theta(quat0)
position0 = position2array(trans_slider.transform.translation)
x_init = np.append(position0[0:2], theta0)

# movable
quat1 = quaternion2array(trans_movable.transform.rotation)
theta1 = quaternion2theta(quat1)
position1 = position2array(trans_movable.transform.translation)
x_movable = np.append(position1[0:2], theta1)

# immovable
quat2 = quaternion2array(trans_immovable1.transform.rotation)
theta2 = quaternion2theta(quat2)
position2 = position2array(trans_immovable1.transform.translation)
x_immovable1 = np.append(position2[0:2], theta2)

quat3 = quaternion2array(trans_immovable2.transform.rotation)
theta3 = quaternion2theta(quat3)
position3 = position2array(trans_immovable2.transform.translation)
x_immovable2 = np.append(position3[0:2], theta3)

x_obs = np.r_[[x_movable], [x_immovable1], [x_immovable2]].tolist()

scene_pkl = {'target': {'x': x_init, 'geom': geom_target},
             'obstacle': {'num': 3,
                          'miu': [miu_movable, miu_immovable, miu_immovable],
                          'geom': [geom_movable, geom_immovable, geom_immovable],
                          'A_list': [compute_limit_surface(geom_movable), compute_limit_surface(geom_immovable), compute_limit_surface(geom_immovable)],
                          'type': [0, 0, 0],  # 0 for immovable, 1 for movable
                          'x': x_obs},
             'contact': {'dt': dt_contact}}

pickle.dump(scene_pkl, open('./planning_scene_robot.pkl', 'wb'))
print(scene_pkl)

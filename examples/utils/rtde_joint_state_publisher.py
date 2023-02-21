import rospy
from rtde_control import RTDEControlInterface as RTDEControl
from sensor_msgs.msg import JointState

def main():
    rate = rospy.Rate(125)
    rtde_c = RTDEControl('192.168.101.50')
    pub = rospy.Publisher('/joint_states', JointState)
    seq_num = 0
    while not rospy.is_shutdown():
        msg = JointState
        msg.header.seq = seq_num
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.name = ['elbow_joint', 
                    'shoulder_lift_joint', 
                    'shoulder_pan_joint', 
                    'wrist_1_joint', 
                    'wrist_2_joint', 
                    'wrist_3_joint']
        msg.position = rtde_c.getActualJointPositionsHistory(0)
        pub.publish(msg)
        seq_num += 1
        rate.sleep()


if __name__ == '__main__':
    nh = rospy.init_node('rtde_utils', anonymous=True)
    main()

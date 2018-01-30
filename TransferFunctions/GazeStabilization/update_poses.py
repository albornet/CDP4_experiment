# Imported Python Transfer Function
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from collections import defaultdict
from geometry_msgs.msg import Pose
import rospy
rospy.wait_for_service("/gazebo/get_link_state")
link_state_proxy_ = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState, persistent=True)


@nrp.MapVariable("link_state_proxy", initial_value=link_state_proxy_)
@nrp.MapVariable("robot_poses", initial_value=defaultdict(lambda: Pose()), scope=nrp.GLOBAL)
@nrp.MapVariable("robot_last_poses", initial_value=defaultdict(lambda: Pose()), scope=nrp.GLOBAL)
@nrp.MapVariable("robot_initial_poses", initial_value=defaultdict(lambda: Pose()), scope=nrp.GLOBAL)
@nrp.MapVariable("joint_states", initial_value=defaultdict(lambda: Pose()), scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", JointState))
@nrp.MapRobotSubscriber("link_states", Topic("/gazebo/link_states", LinkStates))
@nrp.Robot2Neuron()
def update_poses(t, link_state_proxy, robot_poses, robot_last_poses, robot_initial_poses, link_states, joint_states,
                 joints):

    robot_poses = robot_poses.value
    joints = joints.value
    joint_states.value = joints

    link_states = link_states.value
    link_state_proxy = link_state_proxy.value

    robot_head_index = link_states.name.index('robot::head')
    robot_left_eye_index = link_states.name.index('robot::left_eye')
    robot_right_eye_index = link_states.name.index('robot::right_eye')

    robot_last_poses.value = dict(robot_poses)
    # For VCR
    robot_poses['head_abs'] = link_states.pose[robot_head_index]
    # head pose relative to chest frame
    robot_poses['head_rel'] = link_state_proxy('robot::head', 'robot::chest').link_state.pose
    
    # For VOR
    robot_poses['left_eye'] = link_states.pose[robot_left_eye_index]
    robot_poses['right_eye'] = link_states.pose[robot_right_eye_index]
    robot_poses['eye_abs'] = link_states.pose[robot_left_eye_index]

    if t == 0.0:
        robot_initial_poses.value = robot_poses

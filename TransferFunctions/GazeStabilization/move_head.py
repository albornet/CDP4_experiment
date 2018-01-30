from geometry_msgs.msg import Pose
from collections import defaultdict


@nrp.MapRobotPublisher("neck_pitch", Topic('/robot/neck_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("neck_yaw", Topic('/robot/neck_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("neck_roll", Topic('/robot/neck_roll/pos', std_msgs.msg.Float64))
@nrp.MapVariable("robot_poses", scope=nrp.GLOBAL)
@nrp.MapVariable("robot_last_poses", initial_value=defaultdict(lambda: Pose()), scope=nrp.GLOBAL)
@nrp.MapVariable("robot_initial_poses", scope=nrp.GLOBAL)
@nrp.MapVariable("joint_states", scope=nrp.GLOBAL)
@nrp.MapVariable("e_vcr", initial_value=np.array([0, 0, 0]), scope=nrp.GLOBAL)
@nrp.MapVariable("e_vcr_vel", initial_value=np.array([0, 0, 0]), scope=nrp.GLOBAL)
@nrp.Neuron2Robot()
def move_head(t, neck_pitch, neck_yaw, neck_roll, robot_poses, robot_last_poses, robot_initial_poses, joint_states, e_vcr, e_vcr_vel):
    
    import tf
    def quatexp(ori):
        return np.array([math.cos(ori.w), math.sin(ori.w) * ori.x, math.sin(ori.w) * ori.y, math.sin(ori.w) * ori.z])

    efq = tf.transformations.euler_from_quaternion
    rpydict = {}

    # VCR
    kp = {'r': 0.1, 'p': 0.5, 'y': 0.5}
    kd = {'r': 0.5, 'p': 0.5, 'y': 0.5}

    reference_head_abs = robot_initial_poses.value['head_abs'].orientation
    head_abs           =         robot_poses.value['head_abs'].orientation

    try:  # ugly hack for the lack of initial_value for robot_last_poses
        last_head_abs = robot_last_poses.value['head_abs'].orientation
    except KeyError:
        last_head_abs = Pose().orientation

    rpydict['reference_head_abs'] = np.array(efq([reference_head_abs.x, reference_head_abs.y, reference_head_abs.z, reference_head_abs.w]))
    rpydict[          'head_abs'] = np.array(efq([          head_abs.x,           head_abs.y,           head_abs.z,           head_abs.w]))
    rpydict[     'last_head_abs'] = np.array(efq([     last_head_abs.x,      last_head_abs.y,      last_head_abs.z,      last_head_abs.w]))
    rpydict[      'err_head_abs'] = np.array(np.array(rpydict['reference_head_abs'] - rpydict['head_abs']))
    rpydict[      'head_vel_abs'] = (rpydict['head_abs'] - rpydict['last_head_abs'])
    rpydict['  err_head_vel_abs'] = np.array([0, 0, 0])  - rpydict['head_vel_abs']

    e_vcr_r = kp['r']*rpydict['err_head_abs'][0] + kd['r']*rpydict['err_head_vel_abs'][0]
    e_vcr_p = kp['p']*rpydict['err_head_abs'][1] + kd['p']*rpydict['err_head_vel_abs'][1]
    e_vcr_y = kp['y']*rpydict['err_head_abs'][2] + kd['y']*rpydict['err_head_vel_abs'][2]

    torso_roll_idx  = joint_states.value.name.index('torso_roll')
    torso_pitch_idx = joint_states.value.name.index('torso_pitch')
    torso_yaw_idx   = joint_states.value.name.index('torso_yaw')

    neck_roll .send_message(std_msgs.msg.Float64(js.position[torso_roll_idx]  + e_vcr_r))
    neck_pitch.send_message(std_msgs.msg.Float64(js.position[torso_pitch_idx] + e_vcr_p))
    neck_yaw  .send_message(std_msgs.msg.Float64(js.position[torso_yaw_idx]   + e_vcr_y))

    e_vcr.value    = rpydict['err_head_abs']
    e_vcr_vel.value = rpydict['err_head_vel_abs']
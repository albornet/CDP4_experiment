@nrp.MapRobotPublisher("eye_tilt",            Topic('/robot/eye_tilt/pos',    std_msgs.msg.Float64))
@nrp.MapRobotPublisher("eye_version",         Topic('/robot/eye_version/pos', std_msgs.msg.Float64))
@nrp.MapVariable(      "robot_poses",                             scope=nrp.GLOBAL)
@nrp.MapVariable(      "robot_last_poses",                        scope=nrp.GLOBAL)
@nrp.MapVariable(      "robot_initial_poses",                     scope=nrp.GLOBAL)
@nrp.MapVariable(      "joint_states",                            scope=nrp.GLOBAL)
@nrp.MapVariable(      "e_okr",               initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable(      "e_vor",               initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable(      "e_okr_vel",           initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable(      "e_vor_vel",           initial_value=None, scope=nrp.GLOBAL)
@nrp.Neuron2Robot()
def move_eye(t, eye_tilt, eye_version, robot_poses, robot_last_poses, robot_initial_poses, joint_states, e_okr, e_vor, e_okr_vel, e_vor_vel):
    
    # Imports and initialization
    import tf
    from geometry_msgs.msg import Pose
    efq     = tf.transformations.euler_from_quaternion
    rpydict = {}

    # VOR
    kp_vor      = 0.5
    kd_vor      = 0.5
    ref_eye_abs = robot_initial_poses.value['eye_abs'].orientation
    eye_abs     = robot_poses.value['eye_abs'].orientation
    try:
        last_eye_abs = robot_last_poses.value['eye_abs'].orientation
    except KeyError:
        last_eye_abs = Pose().orientation

    rpydict['ref_eye_abs']     = np.array(efq([ ref_eye_abs.x,  ref_eye_abs.y,  ref_eye_abs.z,  ref_eye_abs.w]))
    rpydict['eye_abs']         = np.array(efq([     eye_abs.x,      eye_abs.y,      eye_abs.z,      eye_abs.w]))
    rpydict['last_eye_abs']    = np.array(efq([last_eye_abs.x, last_eye_abs.y, last_eye_abs.z, last_eye_abs.w]))
    rpydict['err_eye_abs']     = np.array(np.array(rpydict['ref_eye_abs'] - rpydict['eye_abs']))
    rpydict['eye_vel_abs']     = (rpydict['eye_abs'] - rpydict['last_eye_abs'])
    rpydict['err_eye_vel_abs'] = np.array([0, 0, 0]) - rpydict['eye_vel_abs']

    e_vor_r = kp_vor * rpydict['err_eye_abs'][0] + kd_vor * rpydict['err_eye_vel_abs'][0]
    e_vor_p = kp_vor * rpydict['err_eye_abs'][1] + kd_vor * rpydict['err_eye_vel_abs'][1]
    e_vor_y = kp_vor * rpydict['err_eye_abs'][2] + kd_vor * rpydict['err_eye_vel_abs'][2]

    # OKR
    kp_okr          = 0.7
    kd_okr          = 0.5
    cmd_vers        = joint_states.value.position[js.name.index('eye_version')] + 0*e_vor_y
    cmd_tilt        = joint_states.value.position[js.name.index('eye_tilt')]    + 0*e_vor_p
    e_vor.value     = rpydict['err_eye_abs']
    e_vor_vel.value = rpydict['err_eye_vel_abs']
    eye_tilt.send_message(std_msgs.msg.Float64(cmd_tilt))
    eye_version.send_message(std_msgs.msg.Float64(cmd_vers))

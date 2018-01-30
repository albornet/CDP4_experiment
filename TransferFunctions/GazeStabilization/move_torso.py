@nrp.MapRobotPublisher("torso_pitch", Topic('/robot/torso_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("torso_roll", Topic('/robot/torso_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("torso_yaw", Topic('/robot/torso_yaw/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def move_torso(t, torso_pitch, torso_roll, torso_yaw):
    import math
    val_r = 0.1 * math.sin(2 * 3.1415 * 0.2 * (t - 2))
    val_p = 0.1 * math.sin(2 * 3.1415 * 0.1 * (t - 2))
    val_y = 0.1 * math.sin(2 * 3.1415 * 0.2 * (t - 2))
    if t >= 2:
        torso_roll.send_message(std_msgs.msg.Float64(val_r))
        torso_pitch.send_message(std_msgs.msg.Float64(val_p))
        torso_yaw.send_message(std_msgs.msg.Float64(val_y))

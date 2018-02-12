# Imported python transfer function that runs the saccade model
@nrp.MapRobotSubscriber("saliencyOutput", Topic("/saliency_map",    sensor_msgs.msg.Image)) # this will make this transfer function able to use the saliency model's output
@nrp.MapRobotPublisher( "eye_tilt", Topic('/robot/eye_tilt/pos',    std_msgs.msg.Float64))  # this will map the eye tilt and version rostopics to 2 variables of this TF
@nrp.MapRobotPublisher( "eye_vers", Topic('/robot/eye_version/pos', std_msgs.msg.Float64))  # to assign values to these rostopics, when updated by your saccade model
@nrp.Neuron2Robot()
def generate_saccades(t, saliencyOutput, eye_tilt, eye_vers):                               # don't forget the mapped parameters names here (+ time)

    # Imports (sometimes, defining them outside the transfer function (e.g. if you need numpy to initialize a decorator thing) is not sufficient, so better define them here again)
    import numpy
    from scipy import misc

    # Use the saliency model to give a target to the saccade model
    if saliencyOutput.value is not None:  # use ".value" to refer to a transfer function parameter! (except for t...)

        # Collect the saliency output ; "0.7" is to resize the saliency output (224*168) to the camera image (320*240)
        saliencyArray = misc.imresize(numpy.mean(CvBridge().imgmsg_to_cv2(saliencyOutput.value, 'rgb8'), axis=2), 1.0/0.7)

        # Compute the saliency max location, in terms of pixel coordinates
        mostSalientPlace.value = numpy.unravel_index(numpy.argmax(saliencyArray), numpy.shape(saliencyArray))  # I always forget ".value" ...
        (targetRow, targetCol) = (mostSalientPlace.value[0]-240/2, mostSalientPlace.value[1]-320/2)            # (0,0 is the center of the image and gaze)

        # Transform the 2D pixel vector into an eye tilt and an eye version
        radPerPix       = 0.006
        targetTiltShift = targetRow*radPerPix
        targetVersShift = targetCol*radPerPix

        # Here, as an example, I make the eyes of the robot move to the saliency peak, at each second. You can use this as an example for what to put in your model!!!
        loopDuration = 1.0                                                                   # in seconds
        timeInLoop   = t-int(t/loopDuration)*loopDuration                                    # how much past the last loop the time is
        if timeInLoop < 0.02 and eye_tilt.value is not None and eye_vers.value is not None:  # the time-step for transfer functions call is 20 ms
            eye_tilt.value = eye_tilt.value + targetTiltShift                                # update with the angular shifts toward the saliency peaks
            eye_vers.value = eye_vers.value + targetVersShift                                # "+=" does not work in the transfer functions !?
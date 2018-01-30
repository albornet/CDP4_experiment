###################################################################################################################
###################################################################################################################
###                                                                                                             ###
###   THE FINAL GOAL OF THIS EXPERIMENT IS TO:                                                                  ###
###   1) DEPLOY THE SEGMENTATION NETOWRK + SIGNALS AROUND SALIENCY MAX                                          ###
###   2) IF AFTER SEGMENTATION, THE OBJECT IS INTERSTING, GENERATE A SACCADE THERE                              ###
###   3) IF NOT INTERESTING, INHIBIT THIS ZONE IN THE SALIENCY MAP AND TRY THE SECOND MOST INTERESTING OBJECT   ###
###   4) GO TO 1) AND GO AGAIN                                                                                  ###
###                                                                                                             ###
###################################################################################################################
###################################################################################################################

# Imported python transfer function that runs the saccade model in concomitence with the saliency model
import numpy
@nrp.MapRobotSubscriber("joints",           Topic("/robot/joints",          sensor_msgs.msg.JointState))  # this gives us access to the joints of the robot (to get their current values)
@nrp.MapRobotSubscriber("saliencyInput",    Topic("/saliency_map",          sensor_msgs.msg.Image))       # this will make this transfer function able to use the saliency model's
@nrp.MapRobotPublisher( "saliencyOutput",   Topic("/saliency_output",       sensor_msgs.msg.Image))       # this will display the saliency map, after computing the inhibition of return
@nrp.MapRobotPublisher( "eyeTilt",          Topic("/robot/eye_tilt/pos",    std_msgs.msg.Float64))        # this will map the eye tilt and version rostopics to 2 variables of this TF
@nrp.MapRobotPublisher( "eyeVers",          Topic("/robot/eye_version/pos", std_msgs.msg.Float64))        # ... to assign values to these rostopics, when updated by your saccade model
@nrp.MapVariable(       "mostSalientPlace", initial_value=numpy.asfarray([0,0]), scope=nrp.GLOBAL)        # this is used to update / keep track of any value throughout the TF's loops
@nrp.MapVariable(       "visitedPlaces",    initial_value=[])                                             # this is used to mimic inhibition of return
@nrp.Neuron2Robot()                                                                                       # ... nrp.GLOBAL would be useful to pass the value to another TF!
def generate_saccades(t, saliencyInput, saliencyOutput, joints, eyeTilt, eyeVers, mostSalientPlace, visitedPlaces):

    # Here, as an example, I make the eyes of the robot slowly move to the saliency peak. You can use this as an example for what to put in your model!!!
    if saliencyInput.value is not None and t > 3:

        # Imports (sometimes, defining them outside the transfer function (e.g. if you need numpy to initialize a decorator thing) is not sufficient, so better define them here again)
        import numpy
        from cv_bridge import CvBridge
        from scipy import misc

        # Parameters
        pixToRad        =  0.006  # ratio determined through trial and error
        inhibitionRange = 30.0    # in pixels  ; for inhibition of return
        inhibitionTau   =  1.0    # in seconds ; decay of the inhibition

        # Collect the current eyes angles
        currEyeTilt = joints.value.position[joints.value.name.index('eye_tilt')]     # get the current eye tilt
        currEyeVers = joints.value.position[joints.value.name.index('eye_version')]  # get the current eye version

        # Collect the saliency output ; "0.7" is to resize the saliency output (168*224) to the camera image (240*320)
        saliencyArray = misc.imresize(numpy.mean(CvBridge().imgmsg_to_cv2(saliencyInput.value, 'rgb8'), axis=2), 1.0/0.7)
        
        # Update the saliency output with inhibition of return (just visited = large inhibition ; visited long ago = small inhibition)          
        for (tilt, vers, timeVisited) in visitedPlaces.value: # SHOULD TAKE HEAD ANGLES INTO ACCOUNT
            (row, col)       = (int((tilt-currEyeTilt)/pixToRad + 240/2), int((vers-currEyeVers)/pixToRad) + 320/2)            
            inhibitionFactor = (1.0-numpy.exp(-(t-timeVisited)/inhibitionTau))
            for i in xrange(240):
                for j in xrange(320):
                    distance = numpy.sqrt(numpy.power(row-i, 2) + numpy.power(col-j, 2))
                    if distance < inhibitionRange:
                        clientLogger.info(str((inhibitionFactor, distance, 2*numpy.arctan(inhibitionFactor*distance))))
                        inhibitionFactor = 2*numpy.arctan(inhibitionFactor*distance)
                        saliencyArray[i,j] = saliencyArray[i,j]*inhibitionFactor  # DISTANCE SHOULD BE A SMOOTHING FACTOR AS WELL (Gaussian like)?!?!?!?!

        # Publish the saliency output, after taking inhibition of return into account
        saliencyRGB  = numpy.dstack((saliencyArray, saliencyArray, saliencyArray))
        messageFrame = CvBridge().cv2_to_imgmsg(saliencyRGB.astype(numpy.uint8), 'rgb8')
        saliencyOutput.send_message(messageFrame)

        # Compute the saliency max location, in terms of pixel coordinates ; sum over time-steps since last saccade
        rawMax                 = numpy.unravel_index(numpy.argmax(saliencyArray), numpy.shape(saliencyArray))  # ".value" for MapRobotSubscribers and MapVariables!
        mostSalientPlace.value = mostSalientPlace.value + numpy.asfarray([rawMax[0]-240/2, rawMax[1]-320/2])   # (0,0 is the center of the image and gaze)

        # If the time has come the eyes move to the target
        saccadeLoopDuration = 0.5                                               # units: seconds
        timeInLoop          = t-int(t/saccadeLoopDuration)*saccadeLoopDuration  # how much past the update loop the time is
        if joints.value is not None and timeInLoop < 0.02:  # the transfer functions are called once every 20 ms of gazebo simulation

            # Transform the 2D pixel vector into an eye tilt and an eye version
            nStepsPerLoop   = saccadeLoopDuration/0.02                                  # to do the mean of the saliency max position
            targetTiltShift = -mostSalientPlace.value[0]*pixToRad/nStepsPerLoop  # direction and amplitude of target row
            targetVersShift = -mostSalientPlace.value[1]*pixToRad/nStepsPerLoop  # direction and amplitude of target column

            # Use the current values of the eyes angles and update them if not too extreme
            if currEyeTilt+targetTiltShift < 0.5 and currEyeVers+targetVersShift < 0.7:
                eyeTilt.send_message(std_msgs.msg.Float64(currEyeTilt+targetTiltShift))  # update with the angular shifts toward the saliency peaks
                eyeVers.send_message(std_msgs.msg.Float64(currEyeVers+targetVersShift))  # "+=" does not work in the transfer functions !?

            # Update the already visited places list
            visitedPlaces.value.append((currEyeTilt+targetTiltShift, currEyeVers+targetVersShift, t))

            # Reset the saliency maximum computation
            mostSalientPlace.value = numpy.asfarray([0,0])

            # # If you want to debug
            # clientLogger.info('Previous angle: '+str((currEyeTilt                , currEyeVers                )))
            # clientLogger.info('Shift    angle: '+str((            targetTiltShift,             targetVersShift)))
            # clientLogger.info('Target   angle: '+str((currEyeTilt+targetTiltShift, currEyeVers+targetVersShift)))
            # clientLogger.info(' ')

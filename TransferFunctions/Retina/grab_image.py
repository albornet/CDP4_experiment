# Imported python transfer function that transforms the camera output into a DC current source fed to LGN neurons
import sensor_msgs.msg
# @nrp.MapRobotSubscriber('saliencyOutput',     Topic('/saliency_map', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber('camera',             Topic('/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher( 'inputToLGN',         Topic('/robot/inputToLGN', sensor_msgs.msg.Image))
@nrp.MapSpikeSource(    'LGNBrightInput',     nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNBright[i]), nrp.dc_source)
@nrp.MapSpikeSource(    'LGNDarkInput',       nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNDark[i]),   nrp.dc_source)
@nrp.MapVariable(       'isBlinking',         scope=nrp.GLOBAL, initial_value=False)
# @nrp.MapVariable(       'mostSalientPlace',   scope=nrp.GLOBAL, initial_value=(int((240-nrp.config.brain_root.imageNumPixelRows)/2*0.7), int((320-nrp.config.brain_root.imageNumPixelColumns)/2*0.7)))
@nrp.Robot2Neuron()
def grab_image(t, camera, inputToLGN, LGNBrightInput, LGNDarkInput, isBlinking): # , saliencyOutput, mostSalientPlace):

    # Take the image from the robot's left eye
    if camera.value is not None:

        # Parameters and imports
        import numpy
        from cv_bridge import CvBridge
        nRows = nrp.config.brain_root.imageNumPixelRows
        nCols = nrp.config.brain_root.imageNumPixelColumns

        # Read the image into an array, mean over 3 colors, resize it for the network and flatten the result
        img = numpy.mean(CvBridge().imgmsg_to_cv2(camera.value, 'rgb8'), axis=2)
        if isBlinking.value:
            img = numpy.zeros(img.shape)

        # # Collect output of the saliency computation model (update only if blinking)
        # if saliencyOutput.value is not None and isBlinking.value:
        #     saliencyOutputArray    = numpy.mean(CvBridge().imgmsg_to_cv2(saliencyOutput.value, 'rgb8'), axis=2)
        #     mostSalientPlace.value = numpy.unravel_index(numpy.argmax(saliencyOutputArray), numpy.shape(saliencyOutputArray))

        # Deploy the segmentation network around the most salient place ("10/7" because saliency map is 168*224)
        firstRow  = int((img.shape[0]-nRows)/2) # max(min(int(1.0/0.7*mostSalientPlace.value[0]) - int(nRows/2), 240-nRows), 0)
        firstCol  = int((img.shape[1]-nCols)/2) # max(min(int(1.0/0.7*mostSalientPlace.value[1]) - int(nCols/2), 320-nCols), 0)
        imgIn     = img[firstRow:firstRow+nRows, firstCol:firstCol+nCols]

        # Show the input image and highlight the input to the LGN with a red rectangle DO IT ON THE SALIENCY MAP AS WELL!!!
        imgInRGB  = numpy.dstack((img, img, img))
        imgInRGB[firstRow      -1,                  firstCol      -1:firstCol+nCols+1, :] = [254.0, 0.0, 0.0]
        imgInRGB[firstRow+nRows+1,                  firstCol      -1:firstCol+nCols+1, :] = [254.0, 0.0, 0.0]
        imgInRGB[firstRow      -1:firstRow+nRows+1, firstCol      -1,                  :] = [254.0, 0.0, 0.0]
        imgInRGB[firstRow      -1:firstRow+nRows+1, firstCol+nCols+1,                  :] = [254.0, 0.0, 0.0]

        # Display the input
        messageFrame = CvBridge().cv2_to_imgmsg(imgInRGB.astype(numpy.uint8), 'rgb8')
        inputToLGN.send_message(messageFrame)

        # Give the segmentation input to the LGN (bright and dark inputs)
        LGNBrightInput.amplitude = 100.0*numpy.maximum(0.0, (imgIn.flatten()/127.0-1.0))
        LGNDarkInput  .amplitude = 100.0*numpy.maximum(0.0, 1.0-(imgIn.flatten()/127.0))

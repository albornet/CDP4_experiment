# Imported python transfer function that transforms the camera output into a DC current source fed to LGN neurons
import sensor_msgs.msg
@nrp.MapRobotSubscriber("camera",         Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("saliencyOutput", Topic("/saliency_map",                         sensor_msgs.msg.Image))
@nrp.MapRobotPublisher( "inputToLGN",     Topic("/robot/inputToLGN",                     sensor_msgs.msg.Image))
@nrp.MapSpikeSource(    "LGNBrightInput", nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNBright[i]), nrp.dc_source)
@nrp.MapSpikeSource(    "LGNDarkInput",   nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNDark[i]),   nrp.dc_source)
@nrp.MapVariable(       "isBlinking",     scope=nrp.GLOBAL, initial_value=False)
@nrp.Robot2Neuron()
def grab_image(t, camera, saliencyOutput, inputToLGN, LGNBrightInput, LGNDarkInput, isBlinking):

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

        # Crop the image (centered patch, using the Laminart network's dimensions)
        firstRow  = int((img.shape[0]-nRows)/2)
        firstCol  = int((img.shape[1]-nCols)/2)
        imgIn     = img[firstRow:firstRow+nRows, firstCol:firstCol+nCols]

        # Add the saliency map to displayed img, if possible
        if saliencyOutput.value is not None:
            from scipy import misc
            saliencyArray  = misc.imresize(numpy.mean(CvBridge().imgmsg_to_cv2(saliencyOutput.value, 'rgb8'), axis=2), 1.0/0.7)
            img            = img + 0.5*saliencyArray
            img[img>254.0] = 254.0

        # Show the input image and highlight the input to the LGN with a red rectangle
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

# Imported python transfer function that transforms the retina output into a DC current source fed to LGN neurons
import sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotSubscriber("image",          Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapRobotPublisher( "inputONToLGN",   Topic("/robot/ganglionON",                     sensor_msgs.msg.Image))
@nrp.MapRobotPublisher( "inputOFFToLGN",  Topic("/robot/ganglionOFF",                    sensor_msgs.msg.Image))
@nrp.MapSpikeSource(    "LGNBrightInput", nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNBright[i]), nrp.dc_source, amplitude=0.0)
@nrp.MapSpikeSource(    "LGNDarkInput",   nrp.map_neurons(range(0, nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.LGNDark[i]),   nrp.dc_source, amplitude=0.0)
@nrp.MapVariable(       "retina",         initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable(       "isBlinking",     initial_value=None, scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def grab_retina_output(t, retina, isBlinking, image, inputONToLGN, inputOFFToLGN, LGNBrightInput, LGNDarkInput):

    # Take the image from the robot's left eye
    if image.value is not None and retina.value is not None:

        # Parameters and imports
        import numpy
        nRows = nrp.config.brain_root.imageNumPixelRows
        nCols = nrp.config.brain_root.imageNumPixelColumns
        retinaToLGNGain = 0.02  # 0.01 is too light

        # Transform the image intput in a retina output
        retina.value.Step(CvBridge().imgmsg_to_cv2(image.value))
        imgON  = numpy.zeros((240, 320))  # this is bad, I should instead feed the retina with zeros ...!
        imgOFF = numpy.zeros((230, 320))  # see 2 lines below for an almost working version
        # if isBlinking.value:
        #     image.value = CvBridge().cv2_to_imgmsg(numpy.zeros(240,320,3))
        if not isBlinking.value:
            imgON  = retina.value.GetOutput("ganglion_bio_ON")
            imgOFF = retina.value.GetOutput("ganglion_bio_OFF")

        # The retina can output weird values just after its initialization
        if (numpy.max(imgON)-numpy.min(imgON)) < 1 or (numpy.max(imgOFF)-numpy.min(imgOFF)) < 1:
            return

        # Take a centered patch of the network's dimensions (either here or below)
        firstRow = int((imgON.shape[0]-nRows)/2)
        firstCol = int((imgON.shape[1]-nCols)/2)
        inputON  = imgON [firstRow:firstRow+nRows, firstCol:firstCol+nCols]
        inputOFF = imgOFF[firstRow:firstRow+nRows, firstCol:firstCol+nCols]
        inputON  = inputON  - numpy.mean(inputON )  # cheat?
        inputOFF = inputOFF - numpy.mean(inputOFF)  # cheat?

        # Give the pre-processed image to the LGN (bright and dark inputs)
        if not numpy.isnan(imgON ).all():
            LGNBrightInput.amplitude = inputON. flatten()*retinaToLGNGain
        if not numpy.isnan(imgOFF).all():
            LGNDarkInput  .amplitude = inputOFF.flatten()*retinaToLGNGain

        # Display the inputs that are given to the LGN cells (input highlighted with a red square)
        imgONRGB  = numpy.dstack((imgON,  imgON,  imgON ))
        imgOFFRGB = numpy.dstack((imgOFF, imgOFF, imgOFF))
        imgONRGB [firstRow,                firstCol:firstCol+nCols, :] = [254.0, 0.0, 0.0]
        imgONRGB [firstRow+nRows,          firstCol:firstCol+nCols, :] = [254.0, 0.0, 0.0]
        imgONRGB [firstRow:firstRow+nRows, firstCol,                :] = [254.0, 0.0, 0.0]
        imgONRGB [firstRow:firstRow+nRows, firstCol+nCols,          :] = [254.0, 0.0, 0.0]
        imgOFFRGB[firstRow,                firstCol:firstCol+nCols, :] = [254.0, 0.0, 0.0]
        imgOFFRGB[firstRow+nRows,          firstCol:firstCol+nCols, :] = [254.0, 0.0, 0.0]
        imgOFFRGB[firstRow:firstRow+nRows, firstCol,                :] = [254.0, 0.0, 0.0]
        imgOFFRGB[firstRow:firstRow+nRows, firstCol+nCols,          :] = [254.0, 0.0, 0.0]
        messageFrameON  = CvBridge().cv2_to_imgmsg(imgONRGB .astype(numpy.uint8), 'rgb8')
        messageFrameOFF = CvBridge().cv2_to_imgmsg(imgOFFRGB.astype(numpy.uint8), 'rgb8')
        inputONToLGN. send_message(messageFrameON )
        inputOFFToLGN.send_message(messageFrameOFF)

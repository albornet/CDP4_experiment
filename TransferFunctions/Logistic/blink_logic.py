# Imported python transfer function that transforms the camera input into a DC current source fed to LGN neurons
import numpy, sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapVariable('isBlinking',         scope=nrp.GLOBAL, initial_value=False)
@nrp.MapVariable('isSegmenting',       scope=nrp.GLOBAL, initial_value=False)
@nrp.Robot2Neuron()
def blink_logic(t, isBlinking, isSegmenting):
    
    # Loop = "nnnsnnnnnbbb", with n -> nothing, s -> segment, b -> blink
    loopDuration       = 0.7
    blinkStart         = 0.6
    segmentationStart  = 0.1

    isBlinking.value   = False
    isSegmenting.value = False

    loopCount    = int(t/loopDuration)
    lastLoopTime = loopCount*loopDuration
    timeInLoop   = t-lastLoopTime

    # Choose if this is a normal or a blinking time-step
    if timeInLoop >= blinkStart:
        isBlinking.value = True
        if timeInLoop < 0.02:
            clientLogger.info('A blinking event happend after %ss of simulation! Duration: %ss...' %(str(t), str(blinkDuration)))

    # Choose whether this is a normal or a segmentation time-step
    if timeInLoop >= segmentationStart and timeInLoop < segmentationStart + 0.02:
        isSegmenting.value = True

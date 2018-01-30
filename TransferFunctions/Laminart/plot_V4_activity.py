# Imported python transfer function that plots the activity of a layer of the model
import sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotPublisher('V4Activity',    Topic('/robot/V4Activity', sensor_msgs.msg.Image))
@nrp.MapSpikeSink(     'V4LayerToPlot', nrp.brain.V4Brightness, nrp.spike_recorder)
@nrp.MapVariable(      'V4Max',         initial_value=0)
@nrp.MapVariable(      'plotSegSignal', scope=nrp.GLOBAL, initial_value=[[[0 for h in range(nrp.config.brain_root.imageNumPixelColumns)] for i in range(nrp.config.brain_root.imageNumPixelRows)] for j in range(nrp.config.brain_root.numSegmentationLayers-1)])
@nrp.Neuron2Robot()
def plot_V4_activity(t, V4LayerToPlot, V4Activity, V4Max, plotSegSignal):

    # Load parameters and libraries
    import numpy
    min_idx             = int(V4LayerToPlot.neurons[0])
    numSegLayers        = nrp.config.brain_root.numSegmentationLayers
    nRows               = nrp.config.brain_root.imageNumPixelRows
    nCols               = nrp.config.brain_root.imageNumPixelColumns
    nNeuronsPerSegLayer = nRows*nCols

    # Build the message to send from the spike recorder
    plotDensityV4 = numpy.zeros(numSegLayers*nNeuronsPerSegLayer)
    msg_to_send   = 254.0*numpy.ones((numSegLayers*(nRows+1)-1,nCols,3), dtype=numpy.uint8)
    for (idx, time) in V4LayerToPlot.times.tolist():
        plotDensityV4[int(idx)-min_idx] = plotDensityV4[int(idx)-min_idx]+1 # +1 spike to the event position
    for h in range(numSegLayers):
        segLayerPlot = plotDensityV4[h*nNeuronsPerSegLayer:(h+1)*nNeuronsPerSegLayer]
        segLayerPlot = numpy.reshape(segLayerPlot, (nRows,nCols))
        if numpy.any(segLayerPlot):
            V4Max.value  = max(numpy.max(segLayerPlot), V4Max.value)
            segLayerPlot = segLayerPlot/V4Max.value*254.0

        # Set up an image to plot V4 neurons activity
        dataV4 = numpy.zeros((nRows,nCols, 3), dtype=numpy.uint8)
        for i in range(nRows):      # Rows
            for j in range(nCols):  # Columns
                dataV4[i][j] = [segLayerPlot[i][j], segLayerPlot[i][j], segLayerPlot[i][j]]

        # Highlight the segmentation signal on the plot
        if h > 0:
            for i in range(nRows):
                for j in range(nCols):
                    if plotSegSignal.value[h-1][i][j] < 0:
                        dataV4[i][j] = [dataV4[i][j][0], dataV4[i][j][1], min(254, dataV4[i][j][2] - plotSegSignal.value[h-1][i][j])]
                    if plotSegSignal.value[h-1][i][j] > 0:
                        dataV4[i][j] = [min(254, dataV4[i][j][0] + plotSegSignal.value[h-1][i][j]/2), min(254, dataV4[i][j][1] + plotSegSignal.value[h-1][i][j]/2), dataV4[i][j][2]]

        # Publish the V4 activity density map
        msg_to_send[h*(nRows+1):(h+1)*(nRows+1)-1,:,:] = dataV4

    # Send the message to display
    msg_frame = CvBridge().cv2_to_imgmsg(msg_to_send.astype(numpy.uint8), 'rgb8')
    V4Activity.send_message(msg_frame)

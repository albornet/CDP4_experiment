# Imported python transfer function that plots the activity of a layer of the model
import sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotPublisher('V2Activity',    Topic('/robot/V2Activity', sensor_msgs.msg.Image))
@nrp.MapSpikeSink(     'V2LayerToPlot', nrp.brain.V2Layer23, nrp.spike_recorder)
@nrp.MapVariable(      'plotSegSignal', scope=nrp.GLOBAL, initial_value=[[[0 for h in range(nrp.config.brain_root.imageNumPixelColumns)] for i in range(nrp.config.brain_root.imageNumPixelRows)] for j in range(nrp.config.brain_root.numSegmentationLayers-1)])
@nrp.Neuron2Robot()
def plot_V2_activity(t, V2LayerToPlot, V2Activity, plotSegSignal):

    # Load parameters and libraries
    import numpy as np
    min_idx             = int(V2LayerToPlot.neurons[0])
    numSegLayers        = nrp.config.brain_root.numSegmentationLayers
    nOris               = nrp.config.brain_root.numOrientations
    nRows               = nrp.config.brain_root.numPixelRows
    nCols               = nrp.config.brain_root.numPixelColumns
    nNeuronsPerSegLayer = nOris*nRows*nCols

    # Build the message to send from the spike recorder
    plotDensityV2 = np.zeros(numSegLayers*nNeuronsPerSegLayer)
    msg_to_send   = 254.0*np.ones((numSegLayers*(nRows+1)-1,nCols,3), dtype=np.uint8)
    for (idx, time) in V2LayerToPlot.times.tolist():
        plotDensityV2[int(idx)-min_idx] = plotDensityV2[int(idx)-min_idx]+1 # +1 spike to the event position
    for h in range(numSegLayers):
        segLayerPlot = plotDensityV2[h*nNeuronsPerSegLayer:(h+1)*nNeuronsPerSegLayer]
        segLayerPlot = np.reshape(segLayerPlot, (nOris,nRows,nCols))
        if np.any(segLayerPlot):
            segLayerPlot = segLayerPlot*254.0

        # Set up a coloured image to plot for V2 oriented neurons activity
        dataV2 = np.zeros((nRows,nCols,3), dtype=np.uint8)
        for i in range(nRows):      # Rows
            for j in range(nCols):  # Columns
                if nOris==2:        # Vertical and horizontal
                    dataV2[i][j] = [segLayerPlot[0][i][j], segLayerPlot[1][i][j], 0]
                if nOris==4:        # Vertical, horizontal, both diagonals
                    diagV2       = max(segLayerPlot[0][i][j], segLayerPlot[2][i][j])
                    dataV2[i][j] = [segLayerPlot[1][i][j], segLayerPlot[3][i][j], diagV2]

        # Highlight the segmentation signal on the plot
        if h > 0:
            for i in range(nRows-1):     # here, nRows is numPixelRows and not imageNumPixelRows (orientation-selective)
                for j in range(nCols-1): # here, nCols is numPixelCols and not imageNumPixelCols (orientation-selective)
                    if plotSegSignal.value[h-1][i][j] < 0:
                        dataV2[i][j] = [dataV2[i][j][0], dataV2[i][j][1], min(254, dataV2[i][j][2] - plotSegSignal.value[h-1][i][j])]
                    if plotSegSignal.value[h-1][i][j] > 0:
                        dataV2[i][j] = [min(254, dataV2[i][j][0] + plotSegSignal.value[h-1][i][j]/2), min(254, dataV2[i][j][1] + plotSegSignal.value[h-1][i][j]/2), dataV2[i][j][2]]

        # Publish the V2 activity density map
        msg_to_send[h*(nRows+1):(h+1)*(nRows+1)-1,:,:] = dataV2

    # Send the message to display
    msg_frame = CvBridge().cv2_to_imgmsg(msg_to_send.astype(np.uint8), 'rgb8')
    V2Activity.send_message(msg_frame)

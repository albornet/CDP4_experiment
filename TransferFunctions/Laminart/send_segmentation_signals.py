import numpy
# Imported python transfer function that sends segmentation signals, following top-down and bottom-up cues (dumb choice of position for now)
@nrp.MapRobotSubscriber("saliencyInput",  Topic("/saliency_map",    sensor_msgs.msg.Image))
# @nrp.MapRobotPublisher( "saliencyOutput", Topic("/saliency_output", sensor_msgs.msg.Image))
@nrp.MapSpikeSource(    "BoundarySegmentationSignalOn", nrp.map_neurons(range(nrp.config.brain_root.useBoundarySegmentation*(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.numPixelRows     *nrp.config.brain_root.numPixelColumns     ), lambda i: nrp.brain.BoundarySegmentationOn[i]), nrp.dc_source, amplitude = 0.0)
@nrp.MapSpikeSource(    "SurfaceSegmentationSignalOn",  nrp.map_neurons(range(nrp.config.brain_root.useSurfaceSegmentation *(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.SurfaceSegmentationOn[i]),  nrp.dc_source, amplitude = 0.0)
@nrp.MapSpikeSource(    "SurfaceSegmentationSignalOff", nrp.map_neurons(range(nrp.config.brain_root.useSurfaceSegmentation *(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.SurfaceSegmentationOff[i]), nrp.dc_source, amplitude = 0.0)
@nrp.MapVariable(       "plotSegSignal",                scope=nrp.GLOBAL, initial_value=numpy.zeros((nrp.config.brain_root.numSegmentationLayers-1, nrp.config.brain_root.imageNumPixelRows, nrp.config.brain_root.imageNumPixelColumns)))
@nrp.MapVariable(       "isBlinking",                   scope=nrp.GLOBAL, initial_value=False)
@nrp.MapVariable(       "isSegmenting",                 scope=nrp.GLOBAL, initial_value=False)
@nrp.Robot2Neuron()
def send_segmentation_signals(t, saliencyInput, BoundarySegmentationSignalOn, SurfaceSegmentationSignalOn, SurfaceSegmentationSignalOff, plotSegSignal, isBlinking, isSegmenting):

    # # Publish the saliency output, after taking inhibition of return into account JUST TO CHECK, REMOVE THIS PARAGRAPH WHEN DONE
    # if saliencyInput.value is not None:
    #     saliencyPlot = 128.0*(1.0+numpy.tanh((misc.imresize(numpy.mean(CvBridge().imgmsg_to_cv2(saliencyInput.value, 'rgb8'), axis=2), 1.0/0.7)-128.0)/30.0))
    #     saliencyRGB  = numpy.dstack((saliencyPlot, saliencyPlot, saliencyPlot))
    #     messageFrame = CvBridge().cv2_to_imgmsg(saliencyRGB.astype(numpy.uint8), 'rgb8')
    #     saliencyOutput.send_message(messageFrame)

    # Loop through all non-basic segmentation layers and send signals around top-down/bottom-up selected targets
    if isSegmenting.value:

        # Imports
        import numpy, random
        from scipy import misc

        # Parameters initialization
        nRows       = nrp.config.brain_root.imageNumPixelRows
        nCols       = nrp.config.brain_root.imageNumPixelColumns
        nSegLayers  = nrp.config.brain_root.numSegmentationLayers
        signalSize  = nrp.config.brain_root.segmentationSignalSize
        firstRow    = int((240-nRows)/2)        # useful for saliency
        firstCol    = int((320-nCols)/2)        # useful for saliency
        sizeOnePart = int(nCols/(nSegLayers-1)) # useful for saliency

        # Segmentation initializatino
        boundaryTargetOn = numpy.zeros(((nRows+1)*(nCols+1)*(nSegLayers-1),))
        surfaceTargetOn  = numpy.zeros(( nRows   * nCols   *(nSegLayers-1),))
        surfaceTargetOff = numpy.zeros(( nRows   * nCols   *(nSegLayers-1),))

        # Collect output of the saliency computation model
        if saliencyInput.value is not None:
            saliencyArray = misc.imresize(numpy.mean(CvBridge().imgmsg_to_cv2(saliencyInput.value, 'rgb8'), axis=2), 1.0/0.7)
            saliencyPart  = 128.0*(1.0+numpy.tanh((saliencyArray[firstRow:firstRow+nRows, firstCol:firstCol+nCols]-128.0)/30.0))

        # Loop for every non basal segmentation layer
        for h in range(nSegLayers-1):

            # Select the target according to the saliency computation
            (segLocRow, segLocCol)     = (int(nRows/2.0), int(nCols/2.0) + int(numpy.power(-1,h)*nCols/4)) # default value
            if saliencyInput.value is not None:
                choiceArray            = numpy.array(range(saliencyPart.size))
                ravelRowCol            = numpy.random.choice(choiceArray, p=numpy.ravel(saliencyPart/float(numpy.sum(saliencyPart))))
                (segLocRow, segLocCol) = numpy.unravel_index(ravelRowCol, saliencyPart.shape)

                # Remove the part of the saliency where a segmentation signal occured, to avoid sending several signals at the same place
                for i in xrange(max(0, segLocRow-signalSize), min(segLocRow+signalSize, nRows)):
                    for j in xrange(max(0, segLocCol-signalSize), min(segLocCol+signalSize, nCols)):
                        distance = numpy.sqrt(numpy.power(segLocRow-i, 2) + numpy.power(segLocCol-j, 2))
                        if distance < 3.0*signalSize:
                            saliencyPart[i,j] = saliencyPart[i,j]*0.5*(1.0+numpy.tanh((distance-2.0*signalSize)/(signalSize/1.5)))

            # Boundary segmentation signal
            if nrp.config.brain_root.useBoundarySegmentation:

                # Tells where a signal has been sent
                clientLogger.info('After '+str(t)+' s of simulation, a boundary segmentation signal has been sent in SL'+str(h+1)+' at location (row, col) = '+str((segLocRow,segLocCol))+'!')
                
                # Set where the segmentation signal is going to be active
                for i in xrange(max(0, segLocRow-signalSize), min(segLocRow+signalSize, nRows+1)):
                    for j in xrange(max(0, segLocCol-signalSize), min(segLocCol+signalSize, nCols+1)):
                        distance = numpy.sqrt(numpy.power(segLocRow-i, 2) + numpy.power(segLocCol-j, 2))
                        if distance < signalSize:
                            boundaryTargetOn[h*(nRows+1)*(nCols+1) + i*(nCols+1) + j] = 10.0
                            plotSegSignal.value[h][min(i, nRows-1)][min(j, nCols-1)] = -254

            # Surface segmentation signal
            if nrp.config.brain_root.useSurfaceSegmentation:

                # Tells where a signal has been sent
                clientLogger.info('After '+str(t)+' s of simulation, a surface segmentation signal has been sent in SL'+str(h+1)+' at location (row, col) = '+str((segLocRow,segLocCol))+'!')
                
                # Set where the segmentation signal is going to be active (OFF signals start from network boundaries)
                for i in xrange(max(0, segLocRow-signalSize), min(segLocRow+signalSize, nRows)):
                    for j in xrange(max(0, segLocCol-signalSize), min(segLocCol+signalSize, nCols)):
                        distance = numpy.sqrt(numpy.power(segLocRow-i, 2) + numpy.power(segLocCol-j, 2))
                        if distance < signalSize:
                            surfaceTargetOn [h*nRows*nCols + i*nCols + j] = 10.0
                            plotSegSignal.value[h][i][j] = 254
                for j in xrange(0, nCols):
                    surfaceTargetOff[h*nRows*nCols + j]                   = 10.0
                    surfaceTargetOff[h*nRows*nCols + (nRows-1)*nCols + j] = 10.0
                for i in xrange(0, nRows):
                    surfaceTargetOff[h*nRows*nCols + i*nCols]             = 10.0
                    surfaceTargetOff[h*nRows*nCols + i*nCols + (nCols-1)] = 10.0

        # Send the segmentation signal for the current segmentation layer
        if nrp.config.brain_root.useBoundarySegmentation:
            BoundarySegmentationSignalOn.amplitude = boundaryTargetOn
        if nrp.config.brain_root.useSurfaceSegmentation:
            SurfaceSegmentationSignalOn .amplitude = surfaceTargetOn
            SurfaceSegmentationSignalOff.amplitude = surfaceTargetOff

    # Reset the segmentation (after a blinking event started)
    if isBlinking.value:
        if nrp.config.brain_root.useBoundarySegmentation:
            BoundarySegmentationSignalOn.amplitude = -10.0
        if nrp.config.brain_root.useSurfaceSegmentation:
            SurfaceSegmentationSignalOn .amplitude = -10.0
            SurfaceSegmentationSignalOff.amplitude = -10.0

    # Update the plot (slow decay)
    plotSegSignal.value = (plotSegSignal.value * 0.8).astype(np.int16)

import numpy
# Imported python transfer function that sends segmentation signals, following top-down and bottom-up cues (dumb choice of position for now)
@nrp.MapSpikeSource("BoundarySegmentationSignalOn", nrp.map_neurons(range(nrp.config.brain_root.useBoundarySegmentation*(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.numPixelRows     *nrp.config.brain_root.numPixelColumns     ), lambda i: nrp.brain.BoundarySegmentationOn[i]), nrp.dc_source, amplitude = 0.0)
@nrp.MapSpikeSource("SurfaceSegmentationSignalOn",  nrp.map_neurons(range(nrp.config.brain_root.useSurfaceSegmentation *(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.SurfaceSegmentationOn[i]),  nrp.dc_source, amplitude = 0.0)
@nrp.MapSpikeSource("SurfaceSegmentationSignalOff", nrp.map_neurons(range(nrp.config.brain_root.useSurfaceSegmentation *(nrp.config.brain_root.numSegmentationLayers-1)*nrp.config.brain_root.imageNumPixelRows*nrp.config.brain_root.imageNumPixelColumns), lambda i: nrp.brain.SurfaceSegmentationOff[i]), nrp.dc_source, amplitude = 0.0)
@nrp.MapVariable(   "plotSegSignal",    scope=nrp.GLOBAL, initial_value=numpy.zeros((nrp.config.brain_root.numSegmentationLayers-1, nrp.config.brain_root.imageNumPixelRows, nrp.config.brain_root.imageNumPixelColumns)))
@nrp.MapVariable(   "isBlinking",       scope=nrp.GLOBAL, initial_value=False)
@nrp.MapVariable(   "isSegmenting",     scope=nrp.GLOBAL, initial_value=False)
@nrp.Robot2Neuron()
def send_segmentation_signals(t, BoundarySegmentationSignalOn, SurfaceSegmentationSignalOn, SurfaceSegmentationSignalOff, plotSegSignal, isBlinking, isSegmenting):
    import random
    import numpy

    # Load parameters
    nRows      = nrp.config.brain_root.imageNumPixelRows      # note: for boundary signal, the correct value is nRows+1
    nCols      = nrp.config.brain_root.imageNumPixelColumns   # note: for boundary signal, the correct value is nCols+1
    nSegLayers = nrp.config.brain_root.numSegmentationLayers
    signalSize = nrp.config.brain_root.segmentationSignalSize

    # Loop through all non-basic segmentation layers and send signals around top-down/bottom-up selected targets
    if isSegmenting.value:
        boundaryTargetOn = numpy.zeros(((nRows+1)*(nCols+1)*(nSegLayers-1),))
        surfaceTargetOn  = numpy.zeros(( nRows   * nCols   *(nSegLayers-1),))
        surfaceTargetOff = numpy.zeros(( nRows   * nCols   *(nSegLayers-1),))

        # Looping, for every non basal segmentation layer, on a small square around the target center
        for h in range(nSegLayers-1):
            
            # Boundary segmentation signal
            if nrp.config.brain_root.useBoundarySegmentation:

                # Choose where a signal is going to start a segmentation
                (segLocRow, segLocCol) = (int((nRows+1)/2.0), int((nCols+1)/2.0) + int(numpy.power(-1,h)*(nCols+1)/4))
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

                # Choose where a signal is going to start a segmentation
                (segLocRow, segLocCol) = (int(nRows/2.0), int(nCols/2.0) + int(numpy.power(-1,h)*nCols/4))
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

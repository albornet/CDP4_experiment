# Code to write a NRP visualization file
import numpy, re

# Parameters initialization
distanceEtalon          = 10
cellType                = None
imageNumPixelRows       = 0
imageNumPixelColumns    = 0
numOrientations         = 0
numSegmentationLayers   = 0
useSurfaceSegmentation  = None
useBoundarySegmentation = None
numBrightnessSpreadingSpeeds  = 0
numSurfaceSegSpreadingSpeeds  = 0
numBoundarySegSpreadingSpeeds = 0
numBrightnessFlows      = 0
numSurfaceSegFlows      = 0
numBoundarySegFlows     = 0

# Run through the brain file to build the ID of all the populations
popID          = []
nNeuronTot     = 0
descriptorKeys = ['nSegs', 'featList', 'nRows', 'nCols']
with open("../../Models/brain_model/visual_segmentation.py", "r") as inputFile:
    for line in inputFile:

        # Look for parameters
        if 'cellType'                      in line and cellType                      == None:
            cellType                       = re.sub('\s+', '', line.split(".")[1].split("(")[0])
        if 'numOrientations'               in line and numOrientations               == 0:
            numOrientations                = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numSegmentationLayers'         in line and numSegmentationLayers         == 0:
            numSegmentationLayers          = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'imageNumPixelRows'             in line and imageNumPixelRows             == 0:
            imageNumPixelRows              = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
            numPixelRows                   = imageNumPixelRows + 1
        if 'imageNumPixelColumns'          in line and imageNumPixelColumns          == 0:
            imageNumPixelColumns           = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
            numPixelColumns                = imageNumPixelColumns + 1
            spaceBetweenSegLayers          = int(imageNumPixelColumns/2) 
        if 'useSurfaceSegmentation'        in line and useSurfaceSegmentation        == None:
            useSurfaceSegmentation         = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'useBoundarySegmentation'       in line and useBoundarySegmentation       == None:
            useBoundarySegmentation        = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numBrightnessSpreadingSpeeds'  in line and numBrightnessSpreadingSpeeds  == 0:
            numBrightnessSpreadingSpeeds   = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numBrightnessFlows'            in line and numBrightnessFlows            == 0:
            numBrightnessFlows             = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numSurfaceSegSpreadingSpeeds'  in line and numSurfaceSegSpreadingSpeeds  == 0:
            numSurfaceSegSpreadingSpeeds   = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numSurfaceSegFlows'            in line and numSurfaceSegFlows            == 0:
            numSurfaceSegFlows             = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numBoundarySegSpreadingSpeeds' in line and numBoundarySegSpreadingSpeeds == 0:
            numBoundarySegSpreadingSpeeds  = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
        if 'numBoundarySegFlows'           in line and numBoundarySegFlows           == 0:
            numBoundarySegFlows            = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))

        # Look for the neuron populations
        if 'sim.Population(' in line:

            # Find the name and the type of the population
            name      =      re.sub('\s+', '', line.split('=')[0])
            typeValue = eval(re.sub('\s+', '', line.split(',')[-1])[:-1])

            # Isolate the descriptors from the nest.Create(...) lines
            descriptorValues = line.split('*')                                                          # descriptors are separated by asterisks
            descriptorValues[ 0] = re.sub('\s+', '', descriptorValues[0].split("(")[-1].split(")")[0])  # remove what's behind the first feature
            descriptorValues[-1] = re.sub('\s+', '', descriptorValues[-1].split(",")[0])                # remove what's after  the last  feature

            # Convert the string-like descriptor values into real values
            descriptorValues = [eval(valString) for valString in descriptorValues]

            # Find the number of neurons in this pop and add it to the total value
            nNeuronPop  = numpy.prod(descriptorValues)
            nNeuronTot += nNeuronPop

            # Take care if several features exist in the population (orientation, flow, etc.), i.e. when there is more than 4 descriptors
            descriptorValues = [descriptorValues[0], list(descriptorValues[1:-2]), descriptorValues[-2], descriptorValues[-1]]

            # Create the descriptor dictionary
            descriptor = {'type': typeValue}
            for index, descriptorKey in enumerate(descriptorKeys):
                descriptor[descriptorKey] = descriptorValues[index]

            # Insert the population as an entry of the popID dictionary
            popID.append((name, descriptor))

# Clean the population list
indicesToKeep = []
for i in range(len(popID)):
    if not('BoundarySegmentation' in popID[i][0] or 'SurfaceSegmentation' in popID[i][0]) \
    or    ('BoundarySegmentation' in popID[i][0] and numSegmentationLayers > 1 and useBoundarySegmentation) \
    or    ('SurfaceSegmentation'  in popID[i][0] and numSegmentationLayers > 1 and useSurfaceSegmentation):
        indicesToKeep.append(i)
popID = [popID[i] for i in indicesToKeep]

# Open and write the file containing the neuron locations
with open("neuronPositions.json", "w") as outputFile:
    outputFile.write('{"populations": {')
    nPopulations = len(popID)
    xDraw = 0 # x-coordinate of the current set of (pop, feature) ; is incremented throughout the loop
    for popIdx in range(nPopulations):

        # Control the population is made of neurons
        popName = popID[popIdx][0]
        thisPop = popID[popIdx][1]
        if thisPop['type'] == 'IF_curr_alpha':

            # Starts writing for this population
            outputFile.write('\n"%s": {"positions": [' % popName)

            # Start to draw the neurons : x = pops, small x = diff. feature types (ori, flow, etc.), y = seg. layers, (z, y) = (row, col)
            totalNumFeatures = numpy.prod(thisPop['featList'])
            for f in range(totalNumFeatures):
                for segLayer in range(thisPop['nSegs']):

                    # Compute y and z coordinate origins
                    yTopLeft = ((thisPop['nCols']+spaceBetweenSegLayers)*int((segLayer+1)/2 + 0.4)*numpy.power(-1, segLayer+1) - int(thisPop['nCols']/2))*distanceEtalon
                    if thisPop['nSegs']%2 == 0:
                        yTopLeft = yTopLeft - int((thisPop['nCols']+spaceBetweenSegLayers)/2)*distanceEtalon
                    zTopLeft = 0 - int(thisPop['nRows']/2)*distanceEtalon

                    # Loop throughout all the neurons of this feature of this segmentation layer of this population
                    for zRow in range(thisPop['nRows']):
                        for yCol in range(thisPop['nCols']):

                            # Write the (x,y,z)-coordinates of this neuron
                            yDraw = yTopLeft + yCol*distanceEtalon
                            zDraw = zTopLeft + zRow*distanceEtalon
                            outputFile.write("[%s, %s, %s]" % (xDraw, yDraw, zDraw))
                            if (f+1)*(segLayer+1)*(zRow+1)*(yCol+1) < numpy.prod(thisPop['featList'])*thisPop['nSegs']*thisPop['nRows']*thisPop['nCols']:
                                outputFile.write(", ")

                xDraw += 1*distanceEtalon  # space on the x-axis between 2 features, for the same population ; formerly 5
            xDraw += 3*distanceEtalon      # space on the x-axis between 2 population                        ; formerly 15

            # Write the end of the line
            if popIdx < nPopulations-1:
                outputFile.write(']},')
            else:
                outputFile.write(']}\n}}')

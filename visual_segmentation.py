#############################################################################
#############################################################################
### This file contains the setup of the neuronal network for segmentation ###
#############################################################################
#############################################################################


from hbp_nrp_cle.brainsim import simulator as sim
from pyNN.connectors import Connector
# from createFilters import createFilters, createPoolingConnectionsAndFilters
import numpy
import nest
import logging
from std_msgs.msg import String

logger = logging.getLogger(__name__)
sim.setup(timestep=1.0, min_delay=1.0, max_delay=2.0, threads=7)


####################################
### Define parameters down here! ###
####################################


# General parameters
imageNumPixelRows    =  40  # number of rows for the network (the input image is scaled to the dimensions of the network)
imageNumPixelColumns = 190  # number of columns for the network (the input image is scaled to the dimensions of the network)
weightScale   = 1.0         # general weight for all connections between neurons
constantInput = 0.5         # input given to the tonic interneuron layer and to the top-down activated segmentation neurons

# Segmentation parameters
numSegmentationLayers        = 3          # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation       = 0          # use segmentation that flows across closed shapes
useBoundarySegmentation      = 1          # use segmentation that flows along connected boundaries
segmentationTargetLocationSD = 5          # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize       = 12         # even number ; circle diameter where a segmentation signal is triggered ; in pixels
if numSegmentationLayers == 1:
    useSurfaceSegmentation   = 0
    useBoundarySegmentation  = 0

# Orientation filters parameters
numOrientations = 2                       # number of orientations   (2, 4 or 8 ; 8 is experimental)
oriFilterSize   = 4                       # better as an even number (how V1 pools from LGN) ; 4 is original
V1PoolSize      = 3                       # better as an odd  number (pooling in V1)         ; 3 is original
V2PoolSize      = 7                       # better as an odd  number (pooling in V2)         ; 7 is original
phi             = 0.0*numpy.pi/2          # filters phase (sometimes useful to shift all orientations)
oppositeOrientationIndex = list(numpy.roll(range(numOrientations), numOrientations/2)) # perpendicular orientations
numPixelRows    = imageNumPixelRows   +1  # number of rows for the oriented grid (in between un-oriented pixels)
numPixelColumns = imageNumPixelColumns+1  # same for columns

# Spreading parameters (V4, surface segmentation, boundary segmentation)
brightnessSpreadingSpeeds              = [1, 2]
surfaceSegmentationSpreadingSpeeds     = [1, 2]
boundarySegmentationSpreadingSpeeds    = [1, 2]
numBrightnessSpreadingSpeeds           = 2 # whatever
numSurfaceSegSpreadingSpeeds           = 2 # whatever
numBoundarySegSpreadingSpeeds          = 2 # whatever
numBrightnessFlows                     = 4 # whatever
numSurfaceSegFlows                     = 2 # whatever
numBoundarySegFlows                    = 2 # whatever

# Neurons parameters
cellParams = {             # parameters for any neuron in the network
    'i_offset'   :   0.0,  # (nA)
    'tau_m'      :  10.0,  # (ms)
    'tau_syn_E'  :   2.0,  # (ms)
    'tau_syn_I'  :   2.0,  # (ms)
    'tau_refrac' :   2.0,  # (ms)
    'v_rest'     : -70.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -56.0,  # (mV) -55.0 is NEST standard, -56.0 good for the current setup
    'cm'         :   0.25} # (nF)
cellType = sim.IF_curr_alpha(**cellParams)

# Connections parameters
connections = {
    # Input and LGN
    'brightInputToLGN'         :   0.1,   # only useful if inputDC == 1
    'darkInputToLGN'           :   0.1,   # only useful if inputDC == 1
    'LGN_ToV1Excite'           :   0.6,   #   400.0,
    'LGN_ToV4Excite'           :   0.2,   #   280.0,

    # V1 layers
    'V1_6To4Excite'            :   0.001, #     1.0,
    'V1_6To4Inhib'             :  -0.001, #    -1.0,
    'V1_4To23Excite'           :   0.5,   #   500.0,
    'V1_23To6Excite'           :   0.1,   #   100.0,
    'V1_ComplexExcite'         :   0.5,   #   500.0,
    'V1_ComplexInhib'          :  -0.5,   #  -500.0,
    'V1_FeedbackExcite'        :   0.5,   #   500.0,
    'V1_NegFeedbackInhib'      :  -1.5,   # -1500.0,
    'V1_InterInhib'            :  -1.5,   # -1500.0,
    'V1_CrossOriInhib'         :  -1.0,   # -1000.0,
    'V1_EndCutExcite'          :   1.5,   #  1500.0,
    'V1_ToV2Excite'            :   10.0,  # 10000.0,

    # V2 layers
    'V2_6To4Excite'            :   0.001, #     1.0,
    'V2_6To4Inhib'             :  -0.02,  #   -20.0,
    'V2_4To23Excite'           :   0.5,   #   500.0,
    'V2_23To6Excite'           :   0.1,   #   100.0,
    'V2_ToV1FeedbackExcite'    :   1.0,   #   (not used at the moment)
    'V2_ComplexExcite'         :   0.5,   #   500.0,
    'V2_ComplexInhib'          :  -1.0,   # -1000.0,
    'V2_ComplexInhib2'         :  -0.1,   #  -100.0,
    'V2_CrossOriInhib'         :  -1.2,   # -1200.0,
    'V2_FeedbackExcite'        :   0.5,   #   500.0,
    'V2_NegFeedbackInhib'      :  -0.8,   #  -800.0,
    'V2_InterInhib'            :  -1.5,   # -1500.0,
    'V2_BoundaryInhib'         : -50.0,   # -5000.0,
    'V2_SegmentInhib'          : -50.0,   #-20000.0, ICI C'ETAIT -20

    # V4 layers
    'V4_BrightnessExcite'      :   5.0,   #  2000.0,
    'V4_BrightnessInhib'       :  -2.5,   # -2000.0,
    'V4_FlowToBrightDarkExcite':   1.0,
    'V4_FlowToBrightDarkInhib' :  -1.0,

    # Surface segmentation layers
    'S_SegmentSpreadExcite'    :   1.0,   #  1000.0,
    'S_SegmentOnOffInhib'      :  -5.0,   # -5000.0,
    'S_SegmentGatingInhib'     :  -5.0,   # -5000.0,

    # Boundary segmentation layers
    'B_SegmentGatingInhib'     :  -25.0,  # -5000.0, ICI C'ETAIT -5
    'B_SegmentSpreadExcite'    :   2.0,   #  2000.0,
    'B_SegmentTonicInhib'      : -20.0,   #-20000.0, ICI C'ETAIT -200
    'B_SegmentOpenFlowInhib'   :  -0.5,   #  -150.0
    'B_TonicInput'             :   0.5}   # DC amplitude on 3rd interneuron for boundary segmentation spreading control

# Scale the weights, if needed
for key, value in connections.items():
    connections[key] = value*weightScale


###############################
### Some useful definitions ###
###############################


# Create a custom connector (use nest.Connect explicitly to go faster)
class MyConnector(Connector):
    def __init__(self, ST):
        self.source = [x[0] for x in ST]
        self.target = [x[1] for x in ST]
    def connect(self, projection):
        nest.Connect([projection.pre.all_cells[s] for s in self.source], [projection.post.all_cells[t] for t in self.target], 'one_to_one', syn_spec=projection.nest_synapse_model)

# Take filter parameters and build 2 oriented filters with different polarities
def createFilters(numOrientations, size, sigma2, Olambda, phi):

    # Initialize the filters
    filters1 = numpy.zeros((numOrientations, size, size))
    filters2 = numpy.zeros((numOrientations, size, size))

    # Fill them with gabors
    midSize = (size-1.)/2.
    maxValue = -1
    for k in range(0, numOrientations):
        theta = numpy.pi*(k+1)/numOrientations + phi
        for i in range(0, size):
            for j in range(0, size):
                x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
                filters1[k][i][j] = numpy.exp(-(x*x + y*y)/(2*sigma2)) * numpy.sin(2*numpy.pi*x/Olambda)
                filters2[k][i][j] = -filters1[k][i][j]

    for k in range(numOrientations):
        sumXX = numpy.sum(filters1[k]*filters1[k])
        filters1[k] /= sumXX
        filters2[k] /= sumXX

    return filters1, filters2

# Take filter parameters and build connection pooling and connection filters arrays
def createPoolingConnectionsAndFilters(numOrientations, VPoolSize, phi):

    # Build the angles in radians, based on the taxi-cab distance
    angles = []
    for k in range(numOrientations/2):
        taxiCabDistance = 4.0*(k+1)/numOrientations
        try:
            alpha = numpy.arctan(taxiCabDistance/(2 - taxiCabDistance))
        except ZeroDivisionError:
            alpha = numpy.pi/2 # because tan(pi/2) = inf
        angles.append(alpha + numpy.pi/4)
        angles.append(alpha - numpy.pi/4)

    # This is kind of a mess, but we could code it better
    for k in range(len(angles)):
        if angles[k] <= 0.0:
            angles[k] += numpy.pi
        if numOrientations == 2: # special case ... but I could not do it otherwise
            angles[k] += numpy.pi/4

    # Sort the angles, because they are generated in a twisted way (hard to explain, but we can see that on skype)
    angles = numpy.sort(angles)

    # Set up orientation kernels for each filter
    midSize = (VPoolSize-1.0)/2.0
    VPoolingFilters = numpy.zeros((numOrientations, VPoolSize, VPoolSize))
    for k in range(0, numOrientations):
        theta = angles[k] + phi
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):

                # Transformed coordinates: rotation by an angle of theta
                x =  (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)

                # If the rotated x value is zero, that means the pixel(i,j) is exactly at the right angle
                if numpy.abs(x) < 0.001:
                    VPoolingFilters[k][i][j] = 1.0

    # Set layer23 pooling connections (connect to points at either extreme of pooling line ; 1 = to the right ; 2 = to the left)
    VPoolingConnections1 = VPoolingFilters.copy()
    VPoolingConnections2 = VPoolingFilters.copy()

    # Do the pooling connections
    for k in range(0, numOrientations):

        # Want only the end points of each filter line (remove all interior points)
        for i in range(1, VPoolSize - 1):
            for j in range(1, VPoolSize - 1):
                VPoolingConnections1[k][i][j] = 0.0
                VPoolingConnections2[k][i][j] = 0.0

        # Segregates between right and left directions
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):
                if j == (VPoolSize-1)/2:
                    VPoolingConnections1[k][0][j] = 0.0
                    VPoolingConnections2[k][VPoolSize-1][j] = 0.0
                elif j < (VPoolSize-1)/2:
                    VPoolingConnections1[k][i][j] = 0.0
                else:
                    VPoolingConnections2[k][i][j] = 0.0

    return VPoolingFilters, VPoolingConnections1, VPoolingConnections2

# Set up filters for Brightness/Darkness/SurfaceSeg/BoundarySeg spreading and boundary blocking
def createBrightnessSurfaceBoundarySpreadingFilters(brightnessSpreadingSpeeds, surfaceSegmentationSpreadingSpeeds, boundarySegmentationSpreadingSpeeds):
    
    # Set up filters for Brightness/Darkness filling-in stage (spreads in various directions) and boundary blocking
    V                             = numOrientations/2-1 # Vertical orientation index
    H                             = numOrientations-1   # Horizontal orientation index
    notStopFlowOrientation        = [H, V]              # Originally had [V, H]
    brightnessFlowFilter          = []
    brightnessBoundaryBlockFilter = []
    for k in range(0, numBrightnessSpreadingSpeeds):
        brightnessFlowFilter.append([[brightnessSpreadingSpeeds[k], 0], [0, brightnessSpreadingSpeeds[k]], [-brightnessSpreadingSpeeds[k], 0], [0, -brightnessSpreadingSpeeds[k]]]) # Right, Down, Left, Up
        brightnessBoundaryBlockFilter.append([])
        directionBBF1 = []
        directionBBF2 = []
        directionBBF3 = []
        directionBBF4 = []
        for d in range(1, (brightnessSpreadingSpeeds[k]+1)): # first index indicates the only orientation that does NOT block flow
            directionBBF1.append([notStopFlowOrientation[1], -d,     0]) # Right
            directionBBF1.append([notStopFlowOrientation[1], -d,    -1]) # Right
            directionBBF2.append([notStopFlowOrientation[0], -1,    -d]) # Down
            directionBBF2.append([notStopFlowOrientation[0],  0,    -d]) # Down
            directionBBF3.append([notStopFlowOrientation[1], (d-1),  0]) # Left
            directionBBF3.append([notStopFlowOrientation[1], (d-1), -1]) # Left
            directionBBF4.append([notStopFlowOrientation[0], -1, (d-1)]) # Up
            directionBBF4.append([notStopFlowOrientation[0],  0, (d-1)]) # Up

        brightnessBoundaryBlockFilter[k].append(directionBBF1)
        brightnessBoundaryBlockFilter[k].append(directionBBF2)
        brightnessBoundaryBlockFilter[k].append(directionBBF3)
        brightnessBoundaryBlockFilter[k].append(directionBBF4)

    # Set up filters for Surface segmentation spreading stage (spreads in various directions) and boundary blocking
    surfaceSegflowFilter          = []  # down, left, up right
    surfaceSegboundaryBlockFilter = []
    for k in range(0, numSurfaceSegSpreadingSpeeds):
        surfaceSegflowFilter.append([[surfaceSegmentationSpreadingSpeeds[k], 0], [0, surfaceSegmentationSpreadingSpeeds[k]]]) # Right, Down
        surfaceSegboundaryBlockFilter.append([])
        directionBBF1= []
        directionBBF2= []
        for d in range(1, (surfaceSegmentationSpreadingSpeeds[k]+1)): # first index indicates the only orientation that does NOT block flow
            directionBBF1.append([notStopFlowOrientation[1], -d,  0]) # Right
            directionBBF1.append([notStopFlowOrientation[1], -d, -1]) # Right
            directionBBF2.append([notStopFlowOrientation[0], -1, -d]) # Down
            directionBBF2.append([notStopFlowOrientation[0],  0, -d]) # Down

        surfaceSegboundaryBlockFilter[k].append(directionBBF1)
        surfaceSegboundaryBlockFilter[k].append(directionBBF2)

    # Set up filters for Boundary segmentation spreading stage (spreads in various directions) and boundary blocking
    boundarySegFlowFilter  = []  # down, left, up right
    for k in range(0, numBoundarySegSpreadingSpeeds):
        boundarySegFlowFilter.append([[boundarySegmentationSpreadingSpeeds[k], 0], [0, boundarySegmentationSpreadingSpeeds[k]]]) # Right, Down

    return brightnessFlowFilter, brightnessBoundaryBlockFilter, surfaceSegflowFilter, surfaceSegboundaryBlockFilter, boundarySegFlowFilter


#########################################################
### Build orientation filters and connection patterns ###
#########################################################


# Set the orientation filters (orientation kernels, V1 and V2 layer23 pooling filters)
filters1, filters2 = createFilters(numOrientations, oriFilterSize, sigma2=0.75, Olambda=4, phi=phi)
V1PoolingFilters, V1PoolingConnections1, V1PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V1PoolSize, phi=phi)
V2PoolingFilters, V2PoolingConnections1, V2PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V2PoolSize, phi=phi)

# Set the filters for Brightness/Darkness/SurfaceSeg/BoundarySeg spreading and boundary blocking
brightnessFlowFilter, brightnessBoundaryBlockFilter, surfaceSegflowFilter, surfaceSegboundaryBlockFilter, boundarySegFlowFilter = createBrightnessSurfaceBoundarySpreadingFilters(brightnessSpreadingSpeeds, surfaceSegmentationSpreadingSpeeds, boundarySegmentationSpreadingSpeeds)


######################################################################################################
### Create the neuron layers ((Retina,) LGN, V1, V2, V4, Boundary and Surface Segmentation Layers) ###
######################################################################################################


# Neural LGN cells will receive input values from LGN
LGNBright = sim.Population(1*1*imageNumPixelRows*imageNumPixelColumns, cellType)
LGNDark   = sim.Population(1*1*imageNumPixelRows*imageNumPixelColumns, cellType)

# Simple oriented neurons
V1Layer6P1 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer6P2 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer4P1 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer4P2 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)

# Complex cells
V1Layer23       = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer23Pool   = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer23Inter1 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer23Inter2 = sim.Population(1*numOrientations*numPixelRows*numPixelColumns, cellType)

###### All subsequent areas have multiple segmentation representations

# Area V2
V2Layer23Inter1 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer23Inter2 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer6        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer4        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer23       = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer23Pool   = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)

# Area V4
V4Brightness         = sim.Population(numSegmentationLayers*1*imageNumPixelRows*imageNumPixelColumns,                                               cellType)
V4BrightnessFlow     = sim.Population(numSegmentationLayers*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
V4InterBrightnessOut = sim.Population(numSegmentationLayers*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
V4Darkness           = sim.Population(numSegmentationLayers*1*imageNumPixelRows*imageNumPixelColumns,                                               cellType)
V4DarknessFlow       = sim.Population(numSegmentationLayers*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
V4InterDarknessOut   = sim.Population(numSegmentationLayers*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns, cellType)

# Segmentation layers
SurfaceSegmentationOn  = []
SurfaceSegmentationOff = []
BoundarySegmentationOn = []
if numSegmentationLayers > 1:

    # Surface Segmentation cells
    if useSurfaceSegmentation == 1:
        SurfaceSegmentationOn        = sim.Population((numSegmentationLayers-1)*1*imageNumPixelRows*imageNumPixelColumns,                                               cellType)
        SurfaceSegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
        SurfaceSegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
        SurfaceSegmentationOff       = sim.Population((numSegmentationLayers-1)*1*imageNumPixelRows*imageNumPixelColumns,                                               cellType)
        SurfaceSegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns, cellType)
        SurfaceSegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns, cellType)

    # Boundary Segmentation cells
    if useBoundarySegmentation == 1:
        BoundarySegmentationOn        = sim.Population((numSegmentationLayers-1)*1*numPixelRows*numPixelColumns,                                                 cellType)
        BoundarySegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numBoundarySegSpreadingSpeeds*numBoundarySegFlows*numPixelRows*numPixelColumns, cellType)
        BoundarySegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numBoundarySegSpreadingSpeeds*numBoundarySegFlows*numPixelRows*numPixelColumns, cellType)
        BoundarySegmentationOnInter3  = sim.Population((numSegmentationLayers-1)*numBoundarySegSpreadingSpeeds*numBoundarySegFlows*numPixelRows*numPixelColumns, cellType)
        ConstantInput                 = sim.DCSource(amplitude=connections['B_TonicInput'], start=0.0, stop=float('inf'))
        ConstantInput.inject_into(BoundarySegmentationOnInter3)


######################################################################
### Neurons layers are defined, now set up connexions between them ###
######################################################################


synapseCount = 0

############### Area V1 ##################

oriFilterWeight = connections['LGN_ToV1Excite']
for k in range(0, numOrientations):                          # Orientations
    for i2 in range(-oriFilterSize/2, oriFilterSize/2):      # Filter rows
        for j2 in range(-oriFilterSize/2, oriFilterSize/2):  # Filter columns
            ST = []                                          # Source-Target vector containing indexes of neurons to connect within specific layers
            ST2 = []                                         # Second Source-Target vector for another connection
            for i in range(oriFilterSize/2, numPixelRows-oriFilterSize/2):         # Rows
                for j in range(oriFilterSize/2, numPixelColumns-oriFilterSize/2):  # Columns
                    if i+i2 >=0 and i+i2<imageNumPixelRows and j+j2>=0 and j+j2<imageNumPixelColumns:
                        # Dark inputs use reverse polarity filter
                        if abs(filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]) > 0.1:
                            ST.append(((i+i2)*imageNumPixelColumns + (j+j2),
                                       k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                        if abs(filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]) > 0.1:
                            ST2.append(((i+i2)*imageNumPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

            if len(ST)>0:
                # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) first polarity filter
                sim.Projection(LGNBright, V1Layer6P1, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer6P2, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNBright, V1Layer4P1, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer4P2, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST)

            if len(ST2)>0:
                # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) second polarity filter
                sim.Projection(LGNBright, V1Layer6P2, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer6P1, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNBright, V1Layer4P2, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer4P1, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST2)

# Excitatory connection from same orientation and polarity 1, input from layer 6
sim.Projection(V1Layer6P1, V1Layer4P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
sim.Projection(V1Layer6P2, V1Layer4P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
synapseCount += (len(V1Layer6P1)+len(V1Layer6P2))

ST = [] # Source-Target vector containing indexes of neurons to connect within specific layers
for k in range(0, numOrientations):          # Orientations
    for i in range(0, numPixelRows):         # Rows
        for j in range(0, numPixelColumns):  # Columns
            for i2 in range(-1,1):
                for j2 in range(-1,1):
                    if i2!=0 or j2!=0:
                        if i+i2 >=0 and i+i2 <numPixelRows and j+j2>=0 and j+j2<numPixelColumns:
                            ST.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                       k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Surround inhibition from layer 6 of same orientation and polarity
sim.Projection(V1Layer6P1, V1Layer4P1, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
sim.Projection(V1Layer6P2, V1Layer4P2, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
synapseCount += 2*len(ST)

ST = []
ST2 = []
ST3 = []
ST4 = []
ST5 = []
ST6 = []
for k in range(0, numOrientations):                                # Orientations
    for i in range(0, numPixelRows):                               # Rows
        for j in range(0, numPixelColumns):                        # Columns
            for k2 in range(0, numOrientations):                   # Other orientations
                if k != k2:
                    ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

            for i2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                for j2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):  # Filter columns

                    if V1PoolingFilters[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                            ST2.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                            ST3.append((oppositeOrientationIndex[k]*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                    if V1PoolingConnections1[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                            ST4.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                    if V1PoolingConnections2[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                            ST5.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

            ST6.append((oppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Layer 4 -> Layer23 (complex cell connections)
sim.Projection(V1Layer4P1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_4To23Excite']))
sim.Projection(V1Layer4P2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_4To23Excite']))
synapseCount += (len(V1Layer4P1)+len(V1Layer4P2))

# Cross-orientation inhibition
sim.Projection(V1Layer23, V1Layer23, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_CrossOriInhib']))
synapseCount += len(ST)

# Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal), V1PoolingFilters pools from both sides
sim.Projection(V1Layer23, V1Layer23Pool, MyConnector(ST2), sim.StaticSynapse(weight=connections['V1_ComplexExcite']))
sim.Projection(V1Layer23, V1Layer23Pool, MyConnector(ST3), sim.StaticSynapse(weight=connections['V1_ComplexInhib']))
synapseCount += (len(ST2) + len(ST3))

# Pooling neurons back to Layer 23 and to interneurons (ST4 for one side and ST5 for the other), V1PoolingConnections pools from only one side
sim.Projection(V1Layer23Pool, V1Layer23,       MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter1, MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter2, MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
sim.Projection(V1Layer23Pool, V1Layer23,       MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter2, MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter1, MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
synapseCount += 3*(len(ST4) + len(ST5))

# Connect interneurons to complex cell and each other
sim.Projection(V1Layer23Inter1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
sim.Projection(V1Layer23Inter2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
synapseCount += (len(V1Layer23Inter1) + len(V1Layer23Inter2))

# End-cutting (excitation from orthogonal interneuron)
sim.Projection(V1Layer23Inter1, V1Layer23, MyConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
sim.Projection(V1Layer23Inter2, V1Layer23, MyConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
synapseCount += 2*len(ST6)

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
sim.Projection(V1Layer23, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
sim.Projection(V1Layer23, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
synapseCount += 2*len(V1Layer23)


############### Area V2  #################

inhibRange64=1
ST = []
ST2 = []
for h in range(0, numSegmentationLayers):        # segmentation layers
    for k in range(0, numOrientations):          # Orientations
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                           h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                for i2 in range(-inhibRange64, inhibRange64+1):
                    for j2 in range(-inhibRange64, inhibRange64+1):
                        if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
                            ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# V2 Layers 4 and 6 connections
sim.Projection(V1Layer23,  V2Layer6, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
sim.Projection(V1Layer23,  V2Layer4, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
sim.Projection(V2Layer6, V2Layer4, sim.OneToOneConnector(),   sim.StaticSynapse(weight=connections['V2_6To4Excite']))
synapseCount += (2*len(ST) + len(V2Layer6))

# Surround inhibition V2 Layer 6 -> 4
sim.Projection(V2Layer6,  V2Layer4, MyConnector(ST2), sim.StaticSynapse(weight=connections['V2_6To4Inhib']))
synapseCount += len(ST2)

ST = []
ST2 = []
ST3 = []
ST4 = []
ST5 = []
ST6 = []
for h in range(0, numSegmentationLayers):        # segmentation layers
    for k in range(0, numOrientations):          # Orientations
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                ST.append((h*numOrientations*numPixelRows*numPixelColumns + oppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                           h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                for i2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                    for j2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):  # Filter columns

                        if V2PoolingFilters[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                                ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                                for k2 in range(0, numOrientations):
                                    if k2 != k:
                                        if k2 == oppositeOrientationIndex[k]:
                                            for h2 in range(0, numSegmentationLayers):  # for all segmentation layers
                                                ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                                        else:
                                            ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                        if V2PoolingConnections1[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                                ST5.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                        if V2PoolingConnections2[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < imageNumPixelRows and j+j2 >= 0 and j+j2 < imageNumPixelColumns:
                                ST6.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

# V2 Layer 4 -> V2 Layer23 (complex cell connections)
sim.Projection(V2Layer4, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_4To23Excite']))
synapseCount += len(V2Layer4)

# Cross-orientation inhibition
sim.Projection(V2Layer23, V2Layer23, MyConnector(ST), sim.StaticSynapse(weight=connections['V2_CrossOriInhib']))
synapseCount += len(ST)

# Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from different + stronger for orthogonal orientation)
sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST2), sim.StaticSynapse(weight=connections['V2_ComplexExcite']))
sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_ComplexInhib']))
synapseCount += (len(ST2) + len(ST3))
if len(ST4)>0:  # non-orthogonal inhibition
    sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_ComplexInhib2']))
    synapseCount += len(ST4)

# Pooling neurons back to Layer 23 and to interneurons (ST5 for one side and ST6 for the other), V2PoolingConnections pools from only one side
sim.Projection(V2Layer23Pool, V2Layer23,       MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter1, MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter2, MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
sim.Projection(V2Layer23Pool, V2Layer23,       MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter2, MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter1, MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
synapseCount += (3*len(ST5) + 3*len(ST6))

# Connect interneurons to complex cell
sim.Projection(V2Layer23Inter1, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
sim.Projection(V2Layer23Inter2, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
synapseCount += (len(V2Layer23Inter1) + len(V2Layer23Inter2))

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
sim.Projection(V2Layer23, V2Layer6, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_23To6Excite']))
synapseCount += len(V2Layer23)

# # Feedback from V2 to V1 (layer 6)
# sim.Projection(V2Layer6, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))
# sim.Projection(V2Layer6, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))


######################## Area V4 filling-in ##################

ST  = []
ST2 = []
ST3 = []
ST4 = []
STX = []
for h in range(numSegmentationLayers):           # Segmentation layers
    for i in range(imageNumPixelRows):           # Rows
        for j in range(imageNumPixelColumns):    # Columns
            for k in range(numBrightnessFlows):  # Flow directions
                for s in range(numBrightnessSpreadingSpeeds): # Set up flow indices for each speed

                    # LGN->V4Flow
                    ST.append((i*imageNumPixelColumns + j, 
                        h*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + s*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j))

                    # Connect V4Flow to V4Brightness/darkness
                    ST2.append((h*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + s*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j, 
                        h*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j))

                    i2 = brightnessFlowFilter[s][k][0]
                    j2 = brightnessFlowFilter[s][k][1]
                    if i + i2 >= 0 and i + i2 < imageNumPixelRows and j + j2 >= 0 and j + j2 < imageNumPixelColumns:
                        ST3.append((h*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + s*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j,
                            h*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + s*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + (i+i2)*imageNumPixelColumns + (j+j2)))

            # # Exploratory thing...
            # for i2 in [0,1]:  # offset by (1,1) to reflect boundary grid is offset from surface grid
            #     for j2 in [0,1]:
            #         if i + i2 >= 0 and i + i2 < imageNumPixelRows and j + j2 >= 0 and j + j2 < imageNumPixelColumns:
            #             STX.append((h*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j, 
            #                 h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

    # Boundary blocking of spreading
    for i in range(numPixelRows):                                                # Rows
        for j in range(numPixelColumns):                                         # Columns
            for s in range(numBrightnessSpreadingSpeeds):
                for k in range(numBrightnessFlows):                              # Flow directions
                    for k2 in range(len(brightnessBoundaryBlockFilter[s][k])):   # Boundary block for given speed
                        for k3 in range(numOrientations):
                            if brightnessBoundaryBlockFilter[s][k][k2][0] != k3: # First index in BBF indicates an orientation that does NOT block
                                i2 = brightnessBoundaryBlockFilter[s][k][k2][1]
                                j2 = brightnessBoundaryBlockFilter[s][k][k2][2]
                                if i+i2>=0 and i + i2 < imageNumPixelRows and j+j2>=0 and j + j2 < imageNumPixelColumns:
                                    ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + i*numPixelColumns + j, 
                                        h*numBrightnessSpreadingSpeeds*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns +  s*numBrightnessFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + (i+i2)*imageNumPixelColumns + (j+j2)))

# LGNBright -> V4BrightnessFlow and LGNDark -> V4DarknessFlow
sim.Projection(LGNBright, V4BrightnessFlow, MyConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
sim.Projection(LGNDark,   V4DarknessFlow,   MyConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
synapseCount += 2*len(ST)

# V4Flow <-> Interneurons
sim.Projection(V4BrightnessFlow, V4InterBrightnessOut, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4DarknessFlow,   V4InterDarknessOut,   sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 2*len(V4BrightnessFlow)

# V4Flow -> V4Brightness/Darkness
sim.Projection(V4BrightnessFlow, V4Brightness, MyConnector(ST2), sim.StaticSynapse(weight=connections['V4_FlowToBrightDarkExcite']))
sim.Projection(V4DarknessFlow,   V4Darkness,   MyConnector(ST2), sim.StaticSynapse(weight=connections['V4_FlowToBrightDarkExcite']))
sim.Projection(V4DarknessFlow,   V4Brightness, MyConnector(ST2), sim.StaticSynapse(weight=connections['V4_FlowToBrightDarkInhib']))
sim.Projection(V4BrightnessFlow, V4Darkness,   MyConnector(ST2), sim.StaticSynapse(weight=connections['V4_FlowToBrightDarkInhib']))
synapseCount += 4*len(ST2)

# V4Brightness neighbors <-> Interneurons
sim.Projection(V4InterBrightnessOut, V4BrightnessFlow, MyConnector(ST3), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4InterDarknessOut,   V4DarknessFlow,   MyConnector(ST3), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 2*len(ST3)

# V2layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
sim.Projection(V2Layer23, V4InterBrightnessOut, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
sim.Projection(V2Layer23, V4InterDarknessOut,   MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
synapseCount += 2*len(ST4)

# Strong inhibition between segmentation layers (WHY TWICE?)
if numSegmentationLayers>1:
    ST = []
    for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                for k2 in range(0, numOrientations):
                    for h2 in range(h, numSegmentationLayers-1):
                        ST.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                   (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

    # Boundaries in lower levels strongly inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
    sim.Projection(V2Layer23, V2Layer4, MyConnector(ST), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
    synapseCount += len(ST)


########### Surface segmentation network ############

if numSegmentationLayers > 1 and useSurfaceSegmentation == 1:

    ST   = []
    ST2  = []
    ST3  = []
    ST4  = []
    for h in range(numSegmentationLayers-1):                           # Segmentation layers
        for i in range(imageNumPixelRows):                             # Rows
            for j in range(imageNumPixelColumns):                      # Columns
                for s in range(numSurfaceSegSpreadingSpeeds): # Set up flow indices for each speed
                    for k in range(numSurfaceSegFlows):                # Flow directions
                        ST.append((h*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j, 
                            h*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns + s*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j))

                        i2 = surfaceSegflowFilter[s][k][0]
                        j2 = surfaceSegflowFilter[s][k][1]
                        if i + i2 >= 0 and i + i2 < imageNumPixelRows and j + j2 >= 0 and j + j2 < imageNumPixelColumns:
                            ST2.append((h*imageNumPixelRows*imageNumPixelColumns + (i+i2)*imageNumPixelColumns + (j+j2),
                                h*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns + s*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j))

                # Surface signals inhibit a range of V2 boundaries at lower levels
                for k2 in range(numOrientations):
                    for h2 in range(h+1):
                        for i2 in range(-2, 4):  # offset by (1,1) to reflect boundary grid is offset from surface grid
                            for j2 in range(-2, 4):
                                if i + i2 >= 0 and i + i2 < imageNumPixelRows and j + j2 >= 0 and j + j2 < imageNumPixelColumns:
                                    ST4.append((h*imageNumPixelRows*imageNumPixelColumns + i*imageNumPixelColumns + j, 
                                        h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

        # Boundary blocking of spreading
        for i in range(numPixelRows):                                              # Rows
            for j in range(numPixelColumns):                                       # Columns
                for s in range(numSurfaceSegSpreadingSpeeds):             # Set up flow indices for each speed
                    for k in range(numSurfaceSegFlows):                            # Flow directions
                        for k2 in range(len(surfaceSegboundaryBlockFilter[s][k])): # Boundary block for given speed
                            for k3 in range(numOrientations):                      # Orientations (for boundary blocking)
                                if surfaceSegboundaryBlockFilter[s][k][k2][0] != k3:  # First index ([0]) in BBF indicates an orientation that does NOT block
                                    i2 = surfaceSegboundaryBlockFilter[s][k][k2][1]
                                    j2 = surfaceSegboundaryBlockFilter[s][k][k2][2]
                                    if i+i2>=0 and i + i2 < imageNumPixelRows and j+j2>=0 and j + j2 < imageNumPixelColumns:
                                        for h2 in range(numSegmentationLayers):    # Use boundaries from all segmentation layers
                                            ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                h*numSurfaceSegSpreadingSpeeds*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns +  s*numSurfaceSegFlows*imageNumPixelRows*imageNumPixelColumns + k*imageNumPixelRows*imageNumPixelColumns + (i+i2)*imageNumPixelColumns + (j+j2)))

    # Off signals inhibit on signals (they can be separated by boundaries)
    sim.Projection(SurfaceSegmentationOff, SurfaceSegmentationOn, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentOnOffInhib']))
    synapseCount += len(SurfaceSegmentationOff)

    # SurfaceSegmentationOn/Off <-> Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter1,  MyConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOn,        MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter1, MyConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOff,       MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    synapseCount += 4*len(ST)

    # SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter2,  MyConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOn,        MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter2, MyConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOff,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    synapseCount += 4*len(ST2)

    # V2Layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
    sim.Projection(V2Layer23, SurfaceSegmentationOnInter1,  MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOnInter2,  MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOffInter1, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOffInter2, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    synapseCount += 4*len(ST3)

    # Segmentation -> V2layer4 and Segmentation -> V2Layer23 (gating ; segmentation signals inhibit boundaries at lower levels)
    sim.Projection(SurfaceSegmentationOn, V2Layer4,  MyConnector(ST4), sim.StaticSynapse(weight=connections['S_SegmentGatingInhib']))
    sim.Projection(SurfaceSegmentationOn, V2Layer23, MyConnector(ST4), sim.StaticSynapse(weight=connections['S_SegmentGatingInhib']))
    synapseCount += 2*len(ST4)


########### Boundary segmentation network ###################

if numSegmentationLayers>1 and useBoundarySegmentation==1:

    ST  = []
    ST2 = []
    ST3 = []
    ST4 = []
    ST5 = []
    for h in range(numSegmentationLayers-1):                            # Segmentation layers (not including baseline layer)
        for i in range(numPixelRows):                                   # Rows
            for j in range(numPixelColumns):                            # Columns
                for s in range(numBoundarySegSpreadingSpeeds): # Set up flow indices for each speed
                    for k in range(numBoundarySegFlows):                # Flow directions

                        # Seg signal flows to and from its associated interneurons for each flow direction and speed
                        ST.append((h*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                            h*numBoundarySegSpreadingSpeeds*numBoundarySegFlows*numPixelRows*numPixelColumns + s*numBoundarySegFlows*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                       
                        # Interneuron flows to and from its corresponding neighbor
                        i2 = boundarySegFlowFilter[s][k][0]
                        j2 = boundarySegFlowFilter[s][k][1]
                        if i+i2 >= 0 and i+i2 < numPixelRows and j+j2 >= 0 and j+j2 < numPixelColumns:
                            ST2.append((h*numBoundarySegSpreadingSpeeds* numBoundarySegFlows*numPixelRows*numPixelColumns + s*numBoundarySegFlows*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j, 
                                h*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                        # Boundaries from all layers inhibit interneuron 3 (all orientations)
                        for h2 in range(numSegmentationLayers):
                            for k2 in range(numOrientations):
                                ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                    h*numBoundarySegSpreadingSpeeds*numBoundarySegFlows*numPixelRows*numPixelColumns +  s*numBoundarySegFlows*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                        
                        # Inhibition from interneurons along path of flow
                        for i3 in range(i2+1):
                            for j3 in range(j2+1):
                                if i+i3 >= 0 and i+i3 < numPixelRows and j+j3 >= 0 and j+j3 < numPixelColumns:
                                    ST4.append((h*numBoundarySegSpreadingSpeeds* numBoundarySegFlows*numPixelRows*numPixelColumns + s*numBoundarySegFlows*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i3)*numPixelColumns + (j+j3),
                                        h*numBoundarySegSpreadingSpeeds* numBoundarySegFlows*numPixelRows*numPixelColumns + s*numBoundarySegFlows*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                # Segmentations inhibit boundaries at lower levels
                for k2 in range(numOrientations):
                    for h2 in range(h+1):
                        ST5.append((h*numPixelRows*numPixelColumns + i*numPixelColumns + j, 
                            h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

    # BoundarySegmentationOn <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2)
    sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter1, MyConnector(ST),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOn,       MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    synapseCount += 2*len(ST)

    # Signal spreads from interneuron to neighbor
    sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOn,       MyConnector(ST2),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter2, MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    synapseCount += 2*len(ST2)

    # V2layer23 -> Segmentation Interneurons (boundaries open flow by inhibiting third interneuron)
    sim.Projection(V2Layer23, BoundarySegmentationOnInter3, MyConnector(ST3), sim.StaticSynapse(weight=connections['B_SegmentOpenFlowInhib']))
    synapseCount += len(ST3)

    # Inhibition from third interneuron (itself inhibited by the presence of a boundary)
    sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, MyConnector(ST4), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
    sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, MyConnector(ST4), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
    synapseCount += 2*len(ST4)

    # BoundarySegmentation -> V2Layer4 and BoundarySegmentation -> V2Layer23 (gating)
    sim.Projection(BoundarySegmentationOn, V2Layer4,  MyConnector(ST5), sim.StaticSynapse(weight=connections['B_SegmentGatingInhib']))
    sim.Projection(BoundarySegmentationOn, V2Layer23, MyConnector(ST5), sim.StaticSynapse(weight=connections['B_SegmentGatingInhib']))
    synapseCount += 2*len(ST5)

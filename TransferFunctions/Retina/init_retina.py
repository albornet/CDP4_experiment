@nrp.MapVariable("retina", initial_value=None, scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def init_retina(t, retina):

	if retina.value is None:

	    ################################
		### DEFINE RETINA PARAMETERS ###
		################################

		# OPL parameters
		sigma_center   = 0.88
		sigma_surround = 2.35
		tau_center     = 10   # ms
		tau_surround   = 10   # ms
		n_center       = 2

		# Slow transient
		lambda_OPL     = 100.0 #1000 # Hz
		w_adap         = 0.5  # 0.8
		tau_adap       = 100  # ms

		# Amacrine parameters (gain control)
		sigma_amacrine = 2.5
		tau_amacrine   = 5.0
		g_BA           = 5.0
		lambda_BA      = 50.0

		# Ganglion parameters (X_cells)
		w_trs          = 0.7
		tau_trs        = 20
		T0_BG          = 80
		lambda_BG      = 150
		v_BG           = 0
		sigma_ganglion = 0

		# Ganglion parameters (Y_cells)
		# w_trs          = 1
		# tau_trs        = 50
		# T0_BG          = 80
		# lambda_BG      = 300
		# v_BG           = 0
		# sigma_ganglion = 0.2

		# Simulation parameters
		import pyretina
		retina_ = pyretina.Retina()
		retina_.TempStep(1)          # simulation step (in ms)
		retina_.PixelsPerDegree(0.5) # pixels per degree of visual angle ; original: 1.0 ; isn't it degree per pixel????
		retina_.DisplayDelay(0)      # display delay
		retina_.DisplayZoom(1)       # display zoom
		retina_.DisplayWindows(3)    # Displays per row
		retina_.Input('camera', {'size': (320, 240)})


		#############################
		### CREATE RETINA NEURONS ###
		#############################

		# Light cones and OPL (outer plexiform layer)
		# retina_.Create('GaussFilter',        'spatial_center',   {'sigma': sigma_center})
		retina_.Create('SpaceVariantGaussFilter', 'spatial_center',   {'sigma': sigma_center, 'K': 0.1, 'R0': 30})
		retina_.Create('LinearFilter',       'tmp_center',       {'type': 'Gamma','tau': tau_center,'n': n_center})
		retina_.Create('GaussFilter',        'spatial_surround', {'sigma': sigma_surround})
		retina_.Create('LinearFilter',       'tmp_surround',     {'type': 'Exp', 'tau': tau_surround})
		retina_.Create('Parrot',             'parrot_OPL',       {'gain': lambda_OPL})
		retina_.Create('Parrot',             'parrot_OPL2',      {'gain': 1.0})
		retina_.Create('HighPass',           'highpass_OPL',     {'w': w_adap, 'tau': tau_adap})

		# Amacrine and bipolar
		retina_.Create('GaussFilter',        'spatial_amacrine', {'sigma': sigma_amacrine})
		retina_.Create('LinearFilter',       'tmp_amacrine',     {'type': 'Exp', 'tau': tau_amacrine})
		retina_.Create('StaticNonLinearity', 'SNL_amacrine',     {'slope': lambda_BA, 'offset': g_BA, 'exponent': 2.0})
		retina_.Create('SingleCompartment',  'SC_bipolar',       {'number_current_ports': 1, 'number_conductance_ports': 1, 'Cm': 1.0, 'Rm': 1.0, 'tau': 0.0, 'E': [0.0]})

		# Ganglion ON-cells
		retina_.Create('Parrot',             'parrot_ON',        {'gain': 1.0})
		retina_.Create('HighPass',           'highpass_trs_ON',  {'w': w_trs, 'tau': tau_trs})
		retina_.Create('Rectification',      'rect_trs_ON',      {'T0': T0_BG, 'lambda': lambda_BG, 'V_th': v_BG})
		retina_.Create('GaussFilter',        'spatial_trs_ON',   {'sigma': sigma_ganglion})

		# Ganglion OFF-cells
		retina_.Create('Parrot',             'parrot_OFF',       {'gain': -1.0}) # OFF inverts current
		retina_.Create('HighPass',           'highpass_trs_OFF', {'w': w_trs, 'tau': tau_trs})
		retina_.Create('Rectification',      'rect_trs_OFF',     {'T0': T0_BG, 'lambda': lambda_BG, 'V_th': v_BG})
		retina_.Create('GaussFilter',        'spatial_trs_OFF',  {'sigma': sigma_ganglion})

		# Output of ganglion cells (gains)
		retina_.Create('Parrot',             'parrot_norm1',     {'gain': 1.0/255.0}) # normalize input luminance
		retina_.Create('Parrot',             'parrot_norm2',     {'gain': 1.0/255.0}) # normalize input luminance
		retina_.Create('Parrot',             'ganglion_bio_ON',  {'gain': 0.02*0.0000000001*10**12}) # output in pA
		retina_.Create('Parrot',             'ganglion_bio_OFF', {'gain': 0.02*0.0000000001*10**12}) # otuput in pA


		#################################
		### DEFINE RETINA CONNECTIONS ###
		#################################

		# Light-cones to OPL (outer plexiform layer)
		retina_.Connect('L_cones',          'parrot_norm1',     'Current')
		retina_.Connect('parrot_norm1',     'spatial_center',   'Current')
		retina_.Connect('spatial_center',   'tmp_center',       'Current')
		retina_.Connect('L_cones',          'parrot_norm2',     'Current')
		retina_.Connect('parrot_norm2',     'spatial_surround', 'Current')
		retina_.Connect('spatial_surround', 'tmp_surround',     'Current')
		retina_.Connect(['spatial_center', '-', 'spatial_surround'], 'parrot_OPL', 'Current')
		retina_.Connect('parrot_OPL',       'highpass_OPL',     'Current')

		# OPL to bipolar cells and amacrine cells
		retina_.Connect('highpass_OPL',     'SC_bipolar',       'Current')
		# retina_.Connect('SC_bipolar',       'spatial_amacrine', 'Current')
		# retina_.Connect('spatial_amacrine', 'tmp_amacrine',     'Current')
		# retina_.Connect('tmp_amacrine',     'SNL_amacrine',     'Current')
		# retina_.Connect('SNL_amacrine',     'SC_bipolar',       'Conductance')
		retina_.Connect('SC_bipolar',       'SNL_amacrine',     'Current')
		retina_.Connect('SNL_amacrine',     'spatial_amacrine', 'Current')
		retina_.Connect('spatial_amacrine', 'tmp_amacrine',     'Current')
		retina_.Connect('tmp_amacrine',     'SC_bipolar',       'Conductance')

		# Bipolar cells to ganglion cells
		retina_.Connect('SC_bipolar',       'parrot_ON',        'Current')
		retina_.Connect('parrot_ON',        'highpass_trs_ON',  'Current')
		retina_.Connect('highpass_trs_ON',  'rect_trs_ON',      'Current')
		retina_.Connect('rect_trs_ON',      'spatial_trs_ON',   'Current')
		retina_.Connect('SC_bipolar',       'parrot_OFF',       'Current')
		retina_.Connect('parrot_OFF',       'highpass_trs_OFF', 'Current')
		retina_.Connect('highpass_trs_OFF', 'rect_trs_OFF',     'Current')
		retina_.Connect('rect_trs_OFF',     'spatial_trs_OFF',  'Current')
		retina_.Connect('spatial_trs_ON',   'ganglion_bio_ON',  'Current')
		retina_.Connect('spatial_trs_OFF',  'ganglion_bio_OFF', 'Current')


		#################################
		### CREATE AND DISPLAY RETINA ###
		#################################

		retina.value = retina_
		retina_.Show('ganglion_bio_ON',  True, {'margin': 0})
		retina_.Show('ganglion_bio_OFF', True, {'margin': 0})

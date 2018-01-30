import hbp_nrp_cle.tf_framework as nrp
@nrp.NeuronMonitor(nrp.brain.V2Layer23, nrp.spike_recorder)
def all_spikes_monitor(t):
	return True
import numpy as np
import qscout

class GatePulse(qscout.v1.std):
	def gate_G(self, qubit):
		return [PulseData(qubit, # output channel
						  1.25e-6, # duration (s)
						  freq0=200e6, # frequency (Hz)
						  amp0=50, # amplitude (arb.)
						  phase0=0)] # phase (deg.)
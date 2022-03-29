import numpy as np
import qscout

class GatePulse(qscout.v1.std):
	def gate_G_coprop(self, qubit):
		return [PulseData(qubit, 1e-6, freq0=200e6, freq1=242e6, fb_enable_mask=0b10)]
	def gate_G_counterprop(self, qubit):
		return [PulseData(qubit, 1e-6, freq0=200e6, fb_enable_mask=0b00),
				PulseData(GLOBAL_BEAM, 1e-6, freq0=242e6, fb_enable_mask=0b01)]
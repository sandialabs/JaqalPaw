import numpy as np
import qscout

class GatePulse(qscout.v1.std):
	def gate_G(self, qubit):
		return [PulseData(qubit, 1.25e-6, amp0=50)]
	def gate_gap(self, qubit):
		return [PulseData(qubit, .25e-6)]
	def gate_G_gap(self, qubit):
		return [PulseData(qubit, 1.25e-6, amp0=50), PulseData(qubit, 0.25e-6)]
	def gate_G_gap_multi(self, qubit, loops):
		return [PulseData(qubit, 1.25e-6, amp0=50), PulseData(qubit, 0.25e-6)] * loops
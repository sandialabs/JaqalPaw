from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins


class Example4GatePulses(QSCOUTBuiltins):
    def gate_G(self, qubit):
        return [PulseData(qubit, 1.25e-6, amp0=50)]

    def gate_gap(self, qubit):
        return [PulseData(qubit, 0.25e-6)]

    def gate_G_gap(self, qubit):
        return [PulseData(qubit, 1.25e-6, amp0=50), PulseData(qubit, 0.25e-6)]

    def gate_G_gap_multi(self, qubit, loops):
        return [PulseData(qubit, 1.25e-6, amp0=50), PulseData(qubit, 0.25e-6)] * loops


class jaqal_pulses:
    GatePulses = Example4GatePulses

from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM


class ExamplesIIIAGatePulses(QSCOUTBuiltins):
    def gate_G5(self, qubit):
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=[10, 30, 20, 50], phase0=0)]

    def gate_G6(self, qubit):
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=(10, 30, 20, 50), phase0=0)]

    def gate_G7(self, qubit):
        spline_amps = (10, 30, 20, 50, 20, 60, 30, 50)
        discrete_amps = [10, 30, 20, 30, 50]
        return [
            PulseData(
                qubit,
                5e-6,
                freq0=200e6,
                freq1=230e6,
                amp0=spline_amps,
                amp1=discrete_amps,
            )
        ]

    def gate_G8(self, qubit):
        return [
            PulseData(qubit, 2e-6, amp0=(0, 9, 41, 50)),
            PulseData(qubit, 2e-6, amp0=50),
            PulseData(qubit, 2e-6, amp0=(50, 0)),
        ]

    def gate_G9(self, qubit):
        return [
            PulseData(qubit, 2e-6, amp0=(0, 9, 41, 50)),
            PulseData(qubit, 2e-6, amp0=50),
            PulseData(qubit, 2e-6, amp0=(50, 0)),
            PulseData(GLOBAL_BEAM, 4.5e-6, amp0=(0, 30, 20, 70)),
        ]  # Fixed typo in docs here.


class jaqal_pulses:
    GatePulses = ExamplesIIIAGatePulses

from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins


class Example3GatePulses(QSCOUTBuiltins):
    def gate_G(self, qubit):
        return [
            PulseData(
                qubit,  # output channel
                1.25e-6,  # duration (s)
                freq0=200e6,  # frequency (Hz)
                amp0=50,  # amplitude (arb.)
                phase0=0,
            )
        ]  # phase (deg.)


class jaqal_pulses:
    GatePulses = Example3GatePulses

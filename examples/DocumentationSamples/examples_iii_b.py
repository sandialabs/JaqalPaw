from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins


class ExamplesIIIBGatePulses(QSCOUTBuiltins):
    def gate_G10A(self, qubit):
        return [PulseData(qubit, 1e-6, framerot0=10) for _ in range(3)]

    def gate_G10B(self, qubit):
        return [PulseData(qubit, 3e-6, framerot0=[10, 10, 10])]

    def gate_G11(self, qubit):
        return [
            PulseData(
                qubit, 1e-6, framerot0=15, fwd_frame0_mask=0b01, inv_frame0_mask=0b00
            ),
            PulseData(
                qubit, 1e-6, framerot0=15, fwd_frame0_mask=0b10, inv_frame0_mask=0b10
            ),
            PulseData(
                qubit, 1e-6, framerot0=15, fwd_frame0_mask=0b11, inv_frame0_mask=0b01
            ),
        ]

    def gate_G12(self, qubit):
        return [
            PulseData(qubit, 1e-6, framerot0=10, apply_at_end_mask=1),
            PulseData(qubit, 1e-6),  # NOP
            PulseData(qubit, 1e-6, framerot0=-5, rst_frame_mask=1),
        ]

    def gate_G13(self, qubit):
        return [
            PulseData(qubit, 1e-6, framerot0=15),
            PulseData(qubit, 3e-6, framerot0=(0, 10, -10, -5)),
            PulseData(qubit, 1e-6),
        ]  # NOP


class jaqal_pulses:
    GatePulses = ExamplesIIIBGatePulses

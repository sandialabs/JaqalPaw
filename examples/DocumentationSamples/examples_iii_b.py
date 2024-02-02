from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins

# Note: Some gate definitions differ from the examples in the manual, in which
# fwd_frame0_mask has been added as an argument. This is because
# fwd_frame0_mask is needed if you want to see the applied phase from the
# frame, instead of the frame's internal value, when using jaqalpaw-emulate.
# Since frame forwarding requirements differ based on the gate type,
# jaqalpaw-emulate displays applied phase during a gate, as it is more useful
# for debugging.

class ExamplesIIIBGatePulses(QSCOUTBuiltins):
    def gate_G10A(self, qubit):
        return [PulseData(qubit, 1e-6, framerot0=10, fwd_frame0_mask=0b01) for _ in range(3)]

    def gate_G10B(self, qubit):
        return [PulseData(qubit, 3e-6, framerot0=[10, 10, 10], fwd_frame0_mask=0b01)]

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
            PulseData(qubit, 1e-6, framerot0=10, apply_at_end_mask=1, fwd_frame0_mask=0b01),
            PulseData(qubit, 1e-6, fwd_frame0_mask=0b01),  # NOP
            PulseData(qubit, 1e-6, framerot0=-5, rst_frame_mask=1, fwd_frame0_mask=0b01),
        ]

    def gate_G13(self, qubit):
        return [
            PulseData(qubit, 1e-6, framerot0=15, fwd_frame0_mask=0b01),
            PulseData(qubit, 3e-6, framerot0=(0, 10, -10, -5), fwd_frame0_mask=0b01),
            PulseData(qubit, 1e-6, fwd_frame0_mask=0b01),
        ]  # NOP


class jaqal_pulses:
    GatePulses = ExamplesIIIBGatePulses

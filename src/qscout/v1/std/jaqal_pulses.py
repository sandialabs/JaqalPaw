from ..QSCOUTBuiltins import *


class GatePulses(
    CalibrationParameters,
    ApparatusParameters,
    UtilityPulses,
    StandardJaqalGates,
    Macros,
):
    # UtilityPulses defines gate_RCoIA, and StandardJaqalGates uses
    # gate_R to provide all the other single-qubit Jaqal Gates.
    def gate_R(self, channel, phase, angle):
        return self.gate_RCoIA(channel, angle=angle, phase=phase)

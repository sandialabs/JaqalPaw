from ..QSCOUTBuiltins import *


class GatePulses(
    CalibrationParameters,
    ApparatusParameters,
    UtilityPulses,
    DynamicalDecouplingGates,
    StandardJaqalGates,
    Macros,
):
    # DynamicalDecouplingGates defines gate_SK1, and StandardJaqalGates uses
    # gate_R to provide all the other single-qubit Jaqal Gates.
    def gate_R(self, channel, phase, angle):
        return self.gate_SK1(channel, angle, phase * 180 / pi)

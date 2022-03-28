from ..QSCOUTBuiltins import *


class GatePulses(
    CalibrationParameters,
    ApparatusParameters,
    UtilityPulses,
    DynamicalDecouplingGates,
    StandardJaqalGates,
    Macros,
):
    def gate_R(self, channel, phase, angle):
        return self.gate_SK1(channel, angle, phase * 180 / pi)

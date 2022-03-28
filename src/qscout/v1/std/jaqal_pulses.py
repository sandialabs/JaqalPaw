from ..QSCOUTBuiltins import *


class GatePulses(
    CalibrationParameters,
    ApparatusParameters,
    UtilityPulses,
    StandardJaqalGates,
    Macros,
):
    def gate_R(self, channel, phase, angle):
        return self.gate_CounterProp(channel, angle=angle, phase=phase)

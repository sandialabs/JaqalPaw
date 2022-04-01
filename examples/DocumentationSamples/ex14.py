from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM
from jaqalpaw.utilities.helper_functions import discretize_frequency


class Example14GatePulses(QSCOUTBuiltins):
    aom_center_frequency: float = 200e6
    effective_qubit_splitting: float = 28.123e6
    motional_mode_frequencies: float = [2.5e6, 2.42e6, 2.38e6]
    amplitude_scaling: float = [0.97, 0.96, 0.95]

    def gate_G(self, qubit1, qubit2):
        global_aom_frequency = (
            self.aom_center_frequency + self.effective_qubit_splitting
        )
        individual_aom_frequency = self.aom_center_frequency
        sb_freq = discretize_frequency(self.motional_mode_frequencies[0])
        qubit_freq = discretize_frequency(global_aom_frequency)
        rsb_freq = qubit_freq - sb_freq
        bsb_freq = qubit_freq + sb_freq
        return [
            PulseData(
                GLOBAL_BEAM,
                100e-6,
                freq0=rsb_freq,
                freq1=bsb_freq,
                fb_enable_mask=0b11,
                sync_mask=0b11,
            ),
            PulseData(qubit1, 100e-6, freq0=individual_aom_frequency, sync_mask=0b01),
            PulseData(qubit1, 100e-6, freq0=individual_aom_frequency, sync_mask=0b01),
        ]


class jaqal_pulses:
    GatePulses = Example14GatePulses

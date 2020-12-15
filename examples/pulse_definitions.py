#from octet.intermediateRepresentations import PulseData, PulseException, Discrete, Spline
import math
from typing import List, NewType
import numpy as np

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.exceptions import PulseException

GLOBAL_BEAM_CHANNEL = 0

Frequency = NewType('Frequency', float)  # Hz
Amplitude = NewType('Amplitude', float)  # Arb
Time = NewType('Time', float)  # seconds
RabiCalibration = NewType('RabiCalibration', float)  # radians/(seconds*Arb**2)


class StandardGatePulses:

    aom_center_frequency: Frequency = 215e6
    adjusted_carrier_splitting: Frequency = 28.42e6
    principal_axis_rotation: float = 45.0

    global_rabi_angle_calibration: RabiCalibration = 42.4242e3

    minimum_pulse_time: Time = 10e-9
    nominal_single_qubit_pi_time: Time = 1e-6  # suggested nominal pi time
    single_qubit_rabi_angle_calibrations: List[RabiCalibration] = [83.23671031e3,  # qubit 0
                                                                   82.15614301e3,  # qubit 1
                                                                   82.13671031e3,  # qubit 2
                                                                   82.16675010e3,  # qubit 3
                                                                   83.26320113e3]  # qubit 4

    stronger_motional_mode_frequencies: List[Frequency] = [2.3e6, 2.35e6, 2.42e6, 2.5e6,  2.6e6]
    weaker_motional_mode_frequencies: List[Frequency] = [2.0e6, 2.05e6, 2.12e6, 2.18e6, 2.26e6]

    stronger_motional_mode_rabi_calibrations_bsb: List[List[RabiCalibration]] = [[20e3, 21e3, 19e3, 23e3, 30e3],  # qubit 0
                                                                                 [21e3, 20e3, 19e3, 21e3, 27e3],  # qubit 1
                                                                                 [22e3, 19e3, 20e3, 22e3, 28e3],  # qubit 2
                                                                                 [21e3, 20e3, 19e3, 21e3, 27e3],  # qubit 3
                                                                                 [20e3, 21e3, 19e3, 23e3, 30e3]]  # qubit 4

    stronger_motional_mode_rabi_calibrations_rsb: List[List[Frequency]] = [[10e3, 11e3, 19e3, 13e3, 30e3],  # qubit 0
                                                                           [11e3, 10e3, 19e3, 11e3, 17e3],  # qubit 1
                                                                           [11e3, 19e3, 10e3, 11e3, 18e3],  # qubit 2
                                                                           [11e3, 10e3, 19e3, 11e3, 17e3],  # qubit 3
                                                                           [10e3, 11e3, 19e3, 13e3, 30e3]]  # qubit 4

    weaker_motional_mode_rabi_calibrations_bsb: List[List[Frequency]] = [[20e3, 21e3, 19e3, 23e3, 30e3],  # qubit 0
                                                                         [21e3, 20e3, 19e3, 21e3, 27e3],  # qubit 1
                                                                         [22e3, 19e3, 20e3, 22e3, 28e3],  # qubit 2
                                                                         [21e3, 20e3, 19e3, 21e3, 27e3],  # qubit 3
                                                                         [20e3, 21e3, 19e3, 23e3, 30e3]]  # qubit 4

    weaker_motional_mode_rabi_calibrations_rsb: List[List[Frequency]] = [[10e3, 11e3, 19e3, 13e3, 30e3],  # qubit 0
                                                                         [11e3, 10e3, 19e3, 11e3, 17e3],  # qubit 1
                                                                         [11e3, 19e3, 10e3, 11e3, 18e3],  # qubit 2
                                                                         [11e3, 10e3, 19e3, 11e3, 17e3],  # qubit 3
                                                                         [10e3, 11e3, 19e3, 13e3, 30e3]]  # qubit 4

    nominal_MS_gate_times: List[List[Time]] = [[0, 200e-6, 240e-6, 210e-6, 220e-6],  # -, MS01, MS02, MS03, MS04
                                               [200e-6, 0, 240e-6, 210e-6, 220e-6],  # MS10, -, MS12, MS13, MS14
                                               [240e-6, 240e-6, 0, 210e-6, 220e-6],  # MS20, MS21, -, MS23, MS24
                                               [240e-6, 210e-6, 210e-6, 0, 220e-6],  # MS30, MS31, MS32, -, MS34
                                               [220e-6, 220e-6, 220e-6, 220e-6, 0]]  # MS40, MS41, MS42, MS43, -

    @staticmethod
    def amplitude_from_rabi_angle(rabi_angle, duration, calibration_factor):
        amp = rabi_angle / np.pi / np.sqrt(calibration_factor * duration) * 100.0
        if amp > 100.0:
            raise PulseException(f"Maximum amplitude exceeded! rabi_angle={rabi_angle}, amp={amp}, duration={duration}, calibration_factor={calibration_factor}")
        return amp

    @staticmethod
    def duration_from_rabi_angle(rabi_angle, amplitude, calibration):
        if amplitude > 100.0:
            raise PulseException("Maximum amplitude exceeded!")
        return rabi_angle / np.pi / (calibration * (amplitude/50.0) ** 2)

    def gate_prepare_all(self, qubit_num=8):
        return [PulseData(ch, 3e-7, waittrig=False) for ch in range(qubit_num)] + \
               [PulseData(ch, 3e-7, waittrig=True) for ch in range(qubit_num)]

    def gate_measure_all(self, qubit_num=8):
        return [PulseData(ch, 3e-7, waittrig=False) for ch in range(qubit_num)]*2

    def gate_R_copropagating_square(self, qubit, theta, phi):
        phase = (phi < 0)*180 + theta/math.pi*180
        return [PulseData(qubit,
                          self.duration_from_rabi_angle(phi, 1.0, self.single_qubit_rabi_angle_calibrations[qubit]),
                          amp0=50.0,
                          amp1=50.0,
                          freq0=self.aom_center_frequency-self.adjusted_carrier_splitting/2,
                          freq1=self.aom_center_frequency+self.adjusted_carrier_splitting/2,
                          phase0=0,
                          phase1=phase,
                          fb_enable_mask=0b01,
                          sync_mask=0b11)]

    def gate_R(self, qubit, theta, phi):
        return self.gate_R_copropagating_square(qubit, theta, phi)

    def gate_Rx(self, qubit, phi):
        return self.gate_R(qubit, 0, phi)

    def gate_Ry(self, qubit, phi):
        return self.gate_R(qubit, math.pi/2, phi)

    def gate_Rz(self, qubit, phi):
        return [PulseData(qubit, self.minimum_pulse_time, framerot1=phi)]

    def gate_Px(self, qubit):
        return self.gate_R(qubit, 0, math.pi)

    def gate_Py(self, qubit):
        return self.gate_R(qubit, math.pi/2, math.pi)

    def gate_Pz(self, qubit):
        return [PulseData(qubit, self.minimum_pulse_time, framerot1=180)]

    def gate_Sx(self, qubit):
        return self.gate_R(qubit, 0, math.pi/2)

    def gate_Sy(self, qubit):
        return self.gate_R(qubit, math.pi/2, math.pi/2)

    def gate_Sz(self, qubit):
        return [PulseData(qubit,    self.minimum_pulse_time, framerot1 = 90)]

    def gate_Sxd(self, qubit):
        return self.gate_R(qubit, math.pi, math.pi/2)

    def gate_Syd(self, qubit):
        return self.gate_R(qubit, 3*math.pi/2, math.pi/2)

    def gate_Szd(self, qubit):
        return [PulseData(qubit, self.minimum_pulse_time, framerot1=-90)]

    def gate_I(self, qubit):
        return [PulseData(qubit, self.nominal_single_qubit_pi_time)]

    def gate_MS(self, qubit1, qubit2, theta, phi):
        phase = (theta < 0)*180 + phi/math.pi*180
        duration_scaling = abs(theta/math.pi*2)
        gate_duration = duration_scaling * self.nominal_MS_gate_times[qubit1][qubit2]

        rsb_amp_qubit1 = self.amplitude_from_rabi_angle(np.pi,
                                                        gate_duration,
                                                        self.stronger_motional_mode_rabi_calibrations_rsb[qubit1][0])
        bsb_amp_qubit1 = self.amplitude_from_rabi_angle(np.pi,
                                                        gate_duration,
                                                        self.stronger_motional_mode_rabi_calibrations_bsb[qubit1][0])
        rsb_amp_qubit2 = self.amplitude_from_rabi_angle(np.pi,
                                                        gate_duration,
                                                        self.stronger_motional_mode_rabi_calibrations_rsb[qubit2][0])
        bsb_amp_qubit2 = self.amplitude_from_rabi_angle(np.pi,
                                                        gate_duration,
                                                        self.stronger_motional_mode_rabi_calibrations_bsb[qubit2][0])
        rsb_freq = self.aom_center_frequency+self.adjusted_carrier_splitting/2-self.stronger_motional_mode_frequencies[0]
        bsb_freq = self.aom_center_frequency+self.adjusted_carrier_splitting/2+self.stronger_motional_mode_frequencies[0]
        return [PulseData(GLOBAL_BEAM_CHANNEL,
                          gate_duration,
                          amp0=100.0,
                          freq0=self.aom_center_frequency-self.adjusted_carrier_splitting/2,
                          phase0=phase,
                          fb_enable_mask=0b01,
                          sync_mask=0b01),
                PulseData(qubit1,
                          gate_duration,
                          amp0=rsb_amp_qubit1,
                          amp1=bsb_amp_qubit1,
                          freq0=rsb_freq,
                          freq1=bsb_freq,
                          fb_enable_mask=0b00,
                          sync_mask=0b11),
                PulseData(qubit2,
                          gate_duration,
                          amp0=rsb_amp_qubit2,
                          amp1=bsb_amp_qubit2,
                          freq0=rsb_freq,
                          freq1=bsb_freq,
                          fb_enable_mask=0b00,
                          sync_mask=0b11),
                ]

    def gate_Sxx(self, qubit1, qubit2):
        return self.gate_MS(qubit1, qubit2, math.pi/2, 0)

from octet.intermediateRepresentations import PulseData, PulseException, Discrete, Spline
import math
from typing import List, NewType
import numpy as np

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
            raise PulseException("Maximum amplitude exceeded!")
        return amp

    @staticmethod
    def duration_from_rabi_angle(rabi_angle, amplitude, calibration):
        if amplitude > 100.0:
            raise PulseException("Maximum amplitude exceeded!")
        return rabi_angle / np.pi / (calibration * (amplitude/50.0) ** 2)

    def gate_prepare_all(self, qubit_num):
        return [PulseData(ch, 3e-7, waittrig=False) for ch in range(qubit_num)] + \
               [PulseData(ch, 3e-7, waittrig=True) for ch in range(qubit_num)]

    def gate_measure_all(self, qubit_num):
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
        return [PulseData(qubit, self.minimum_pulse_time, framerot1=90)]

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

    def gate_GaussPLE2(self, qubit, theta):
        phsarray = np.array([2.03684916,  0.60865326,  0.47231682,  1.76940918,  -1.43574221,
                             1.69356529, -0.76578387, -1.36197437, -1.54918721,  1.14904740,
                             2.33834424, -0.57373570,  1.23415720,  0.01946254,  0.16637229])
        return [PulseData(qubit,
                          self.nominal_single_qubit_pi_time*9,
                          freq0=self.aom_center_frequency - self.adjusted_carrier_splitting,
                          freq1=self.aom_center_frequency + self.adjusted_carrier_splitting,
                          amp0=(0.08200746635147083, 4.249905628536256, 45.40407387272451, 100.0,
                                45.404073872724524, 4.249905628536264, 0.08200746635147083),
                          amp1=(0.08200746635147083, 4.249905628536256, 45.40407387272451, 100.0,
                                45.404073872724524, 4.249905628536264, 0.08200746635147083),
                          phase1=tuple(phsarray*180/np.pi+theta),
                          phase0=0,
                          framerot0=0,
                          framerot1=0,
                          sync_mask=3,
                          enable_mask=0,
                          waittrig=False)]


class CustomGateDefinitions(StandardGatePulses):

    def gate_squigglyStraight(self, qubit):
        return [PulseData(qubit,
                          self.nominal_single_qubit_pi_time*10,
                          amp0=(0, 50, 30, 80, 50, 20),
                          amp1=[0, 50, 30, 80, 50, 20],
                          phase0=0,
                          phase1=[50, 30, 0],
                          freq0=(200, 205, 199, 202, 200.1234567),
                          freq1=(200, 228),
                          framerot0=(0, 12, 50, 100, 90),
                          framerot1=[15, 15, 15, 15, 15, 15],
                          ),
                PulseData(qubit,
                          self.nominal_single_qubit_pi_time*10,
                          amp0=(20, 50),
                          amp1=(0, 50, 30, 80, 50, 20),
                          phase0=(0, -20, 50),
                          phase1=(0, 30, 50),
                          freq0=200.1234567,
                          freq1=(228, 200),
                          #framerot0=0,
                          #framerot1=0,
                          ),
                PulseData(qubit,
                          self.nominal_single_qubit_pi_time*10,
                          amp0=(0, 50, 30, 80, 50, 20),
                          amp1=[0, 50, 30, 80, 50, 20],
                          phase0=50,
                          phase1=[50, 30, 50],
                          freq0=(200, 205, 199, 202, 200.1234567),
                          freq1=(200, 228),
                          framerot0=(0, 12, 50, 100, 90),
                          framerot1=(0, 10, -40, -78, -90),
                          )]

    def wave_helper(self, nperiods, pts_per_period=4, amp=50, offset=0):
        xpoints = np.linspace(0,nperiods*2*np.pi, pts_per_period*nperiods)
        ypoints = 50*np.sin(xpoints)+offset
        return ypoints

    def gate_weird(self, qubit, nperiods, square=0):
        conv = list if square else tuple
        return [PulseData(qubit,
                          self.nominal_single_qubit_pi_time*10,
                          amp0=conv(self.wave_helper(nperiods)),
                          amp1=conv(self.wave_helper(nperiods)))]
  
class BrandonPulses(StandardGatePulses):

    def gate_robust_ms(self,qubit1,qubit2):

        # Rabi rates in Hz
        rabis = [0.0,
                 276936.8015078513,
                 141450.5790117699,
                 276907.9261594575,
                 141450.5790117699,
                 276936.8015078513,
                 0.0]

        # detunings from carrier transition in Hz
        detunings = [2853378.6817786656,
                     2999558.960721427,
                     3021082.753418489,
                     3107568.608853587,
                     3063522.493598082,
                     3032234.836329433,
                     3156834.7763933865,
                     2967680.023122517,
                     3107769.453822267,
                     3092894.1128954873,
                     3035625.301950426,
                     3096764.933502942,
                     3038549.749174705,
                     3035004.3970164433,
                     3020346.389851864,
                     3035004.3970164433,
                     3038549.749174705,
                     3096764.933502942,
                     3035625.301950426,
                     3092894.1128954873,
                     3107769.453822267,
                     2967680.023122517,
                     3156834.7763933865,
                     3032234.836329433,
                     3063522.493598082,
                     3107568.608853587,
                     3021082.753418489,
                     2999558.960721427,
                     2853378.6817786656] 

        global_freq = self.aom_center_frequency - self.adjusted_carrier_splitting/2.
       
        bsb_freq = [self.aom_center_frequency
                   + self.adjusted_carrier_splitting/2.
                   + det
                       for det in detunings]

        rsb_freq = [self.aom_center_frequency
                     + self.adjusted_carrier_splitting/2.
                     - det
                         for det in detunings]

        fac_qubit1 = self.single_qubit_rabi_angle_calibrations[qubit1]
        fac_qubit2 = self.single_qubit_rabi_angle_calibrations[qubit2]

        amps_qubit1 = [self.amplitude_from_rabi_rate(rabi,fac_qubit1)
                            for rabi in rabis]  
        amps_qubit2 = [self.amplitude_from_rabi_rate(rabi,fac_qubit2)
                            for rabi in rabis]
        
        return [PulseData(GLOBAL_BEAM_CHANNEL,
                          90e-6,
                          freq0=global_freq,
                          amp0=tuple(amps_qubit1),
                          ),
                PulseData(qubit1,
                          90e-6,
                          freq0=tuple(rsb_freq),
                          amp0=tuple(amps_qubit1),
                          freq1=tuple(bsb_freq),
                          amp1=tuple(amps_qubit1),
                          ),
                PulseData(qubit2,
                          90e-6,
                          freq0=tuple(rsb_freq),
                          amp0=tuple(amps_qubit2),
                          freq1=tuple(bsb_freq),
                          amp1=tuple(amps_qubit2),
                          )]

    def gate_pst_loop(self,qubit):

        # Rabi rates in Hz
        rabis = [0.0,
                 391647.78061262943,
                 200041.32724397225,
                 391606.9447033127,
                 200041.32724397225,
                 391647.78061262943,
                 0.0]

        # detunings from carrier transition in Hz
        detunings = [2853378.6817786656,
                     2999558.960721427,
                     3021082.753418489,
                     3107568.608853587,
                     3063522.493598082,
                     3032234.836329433,
                     3156834.7763933865,
                     2967680.023122517,
                     3107769.453822267,
                     3092894.1128954873,
                     3035625.301950426,
                     3096764.933502942,
                     3038549.749174705,
                     3035004.3970164433,
                     3020346.389851864,
                     3035004.3970164433,
                     3038549.749174705,
                     3096764.933502942,
                     3035625.301950426,
                     3092894.1128954873,
                     3107769.453822267,
                     2967680.023122517,
                     3156834.7763933865,
                     3032234.836329433,
                     3063522.493598082,
                     3107568.608853587,
                     3021082.753418489,
                     2999558.960721427,
                     2853378.6817786656] 

        freq0 = [ self.aom_center_frequency
                - self.adjusted_carrier_splitting/2.]*3
       
        freq1 = [ self.aom_center_frequency
                + self.adjusted_carrier_splitting/2.
                + det
                    for det in detunings] 

        fac = self.single_qubit_rabi_angle_calibrations[0]
        amps = [ self.amplitude_from_rabi_rate(rabi,fac)
                    for rabi in rabis]
        
        return [PulseData(qubit,
                          90e-6,
                          freq0=freq0,
                          amp0=tuple(amps),
                          freq1=tuple(freq1),
                          amp1=tuple(amps),
                          )]

    def amplitude_from_rabi_rate(self, rabi_rate, calibration_factor):
        """Convert from a Rabi rate of a Raman transition to
        to the "amplitude" of one Raman tone. Here, "amplitude" is the percentage
        of the electric field amplitude of one Raman tone used during calibration,
        assuming the calibration used square pulses of equal amplitude for each
        Raman tone. The current Raman transition must also use Raman tones of equal
        amplitude, but a squre pulse is not necessary.
        
        Inputs: rabi_rate = Rabi rate of Raman transition (Hz)

                calibration_factor = rabi_cal_1 * rabi_cal_2 / (2 Delta pi)

        Outputs: amplitude = percentage of the electric field amplitude of one
                             Raman tone used during calibration (%)
        """
        import numpy as np
        return np.sqrt(100*(2*np.pi*rabi_rate)/np.pi/calibration_factor)




       

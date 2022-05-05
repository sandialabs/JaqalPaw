from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
import numpy as np
from scipy.special import erf as _erf

# Either of these import methods work at the moment, but local doesn't work well.
import sys
import os
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM, both_tones, tone0


class HelperFunctions:
    @staticmethod
    def gauss(npoints, A, freqwidth=300e3, total_duration=4e-6):
        trange = np.linspace(-total_duration / 2, total_duration / 2, npoints)
        sigma = 1 / (2 * np.pi * freqwidth)
        return A * np.exp(-(trange**2) / 2 / sigma**2)

    @staticmethod
    def erf(npoints, A, freqwidth=300e3, total_duration=4e-6):
        tdata = np.linspace(-total_duration / 2, total_duration / 2, npoints)
        return (0.5 + _erf(tdata * 2 * np.pi * freqwidth / np.sqrt(2)) / 2) * A

    @staticmethod
    def global_amp_cal(angle):
        # Calibration parameters from varying Rabi rates as a function of global power
        a = 0.0672677
        b = 0.0083695
        omega_pi2 = 0.04975888
        return np.arcsin((omega_pi2 / a) * np.sqrt(abs(angle) / (np.pi / 2))) / (
            b * 100
        )


class MSExemplar(QSCOUTBuiltins, HelperFunctions):

    MS_phi: float = 0.0

    def gate_MS(
        self, channel1, channel2, axis_rad=0, angle=np.pi / 2, is_gaussian=True
    ):

        # First discretize the frequencies to ensure frequency matching and prevent slow phase drift
        MS_freq0_total_corr = (
            discretize_frequency(self.ia_center_frequency)
            + discretize_frequency(self.lower_motional_mode_frequencies[1])
            + discretize_frequency(self.MS_delta)
        )  # Blue (lowest frequency mode)
        MS_freq1_total_corr = (
            discretize_frequency(self.ia_center_frequency)
            - discretize_frequency(self.lower_motional_mode_frequencies[1])
            - discretize_frequency(self.MS_delta)
        )  # Red (lowest frequency mode)

        # Use global beam as the lower leg of the Raman transition.
        global_beam_frequency = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)

        # Convert Jaqal input axis from radians to degrees as the hardware requires degrees
        # Note: for integration with single qubit gates, this defaults to zero and the MS gate phase is determined by an external set of gates (see third exemplar)
        axis = axis_rad * 180 / np.pi

        # If requested MS gate angle is negative, then one ion in the pair will be performed 180 degrees out of phase to get the desired X,-X interaction to create a negative MS gate angle
        phase_add = 180 if angle < 0 else 0

        # Global beam AOM distortion (note: this assumes the global AOM power input is 100 for a pi/2 MS gate (XX))
        global_amp_scale = self.global_amp_cal(angle)
        global_amp = self.amp0_counterprop

        # Define Gaussian pulse shape
        gauss_amp = np.sqrt(self.gauss(13, 1, freqwidth=300e3, total_duration=4e-6))

        # Boolean to dictate whether to use a Gaussian shape, and if Gaussian will automatically apply an error function shaping to the frame rotation throughout the pulse
        # Gaussian or not
        if is_gaussian:
            amp_input = gauss_amp  # Gaussian pulse
            framerot_input = tuple(
                self.erf(
                    13,
                    self.MS_framerot * ((global_amp_scale) ** 3),
                    freqwidth=300e3,
                    total_duration=4e-6,
                )
            )
            framerot_app = 0  # Applies Stark shift correction during pulse

        else:
            amp_input = np.array([1, 1])  # square pulse
            framerot_input = self.MS_framerot
            framerot_app = (
                0b01  # Applies entire Stark shift correction at end of the pulse
            )

        listtoReturn = [
            PulseData(
                GLOBAL_BEAM,
                self.MS_pulse_duration,
                freq0=global_beam_frequency,
                amp0=tuple(amp_input * global_amp * global_amp_scale),
                phase0=0,
                phase1=0,
                sync_mask=both_tones,
                fb_enable_mask=tone0,
            ),
            PulseData(
                channel1,
                self.MS_pulse_duration,
                freq0=MS_freq0_total_corr,
                freq1=MS_freq1_total_corr,
                amp0=tuple(amp_input * self.MS_blue_amp_list[channel1]),
                amp1=tuple(amp_input * self.MS_red_amp_list[channel1]),
                phase0=self.MS_phi + axis,
                phase1=self.MS_phi + axis,
                framerot0=framerot_input,
                fwd_frame0_mask=both_tones,
                apply_at_end_mask=framerot_app,
                sync_mask=both_tones,
                fb_enable_mask=0,
            ),
            PulseData(
                channel2,
                self.MS_pulse_duration,
                freq0=MS_freq0_total_corr,
                freq1=MS_freq1_total_corr,
                amp0=tuple(amp_input * self.MS_blue_amp_list[channel2]),
                amp1=tuple(amp_input * self.MS_red_amp_list[channel2]),
                phase0=self.MS_phi + axis + phase_add,
                phase1=self.MS_phi + axis + phase_add,
                framerot0=framerot_input,
                fwd_frame0_mask=both_tones,
                apply_at_end_mask=framerot_app,
                sync_mask=both_tones,
                fb_enable_mask=0,
            ),
        ]
        return listtoReturn


class jaqal_pulses:
    GatePulses = MSExemplar

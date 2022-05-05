##########   JaqalPaw Exemplar : Modulated MS Gate : SAND Number: SAND2021-11216 O ##########
""" This JaqalPaw code is used to generate frequency, amplitude, and phase
modulated gates. This code was developed in the context of optimal control work
on QSCOUT, and now serves as an example (and potentially base code) for user
pulse definitions where modulated drive pulses are desired.
    Posting: https://qscout.sandia.gov
    Last update: September 14, 2021
    Contact author: Matthew Chow : mnchow@sandia.gov
"""

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
import numpy as np
from scipy.special import erf as _erf
import os
import sys

from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM, both_tones


class HelperFunctions:
    """Collection of staticmethods that are helpful in construction of the
    main pulse definition. This class will be inherited by ModulatedMSExemplar.
    """

    @staticmethod
    def gauss(npoints, A, freqwidth=300e3, total_duration=4e-6):
        """Gives evenly spaced points along a Gaussian as a numpy array."""
        trange = np.linspace(-total_duration / 2, total_duration / 2, npoints)
        sigma = 1 / (2 * np.pi * freqwidth)
        return A * np.exp(-(trange**2) / 2 / sigma**2)

    @staticmethod
    def erf(npoints, A, freqwidth=300e3, total_duration=4e-6):
        """Gives evenly spaced points along an erf function as a numpy array."""
        tdata = np.linspace(-total_duration / 2, total_duration / 2, npoints)
        return (0.5 + _erf(tdata * 2 * np.pi * freqwidth / np.sqrt(2)) / 2) * A

    @staticmethod
    def get_cfg(cfg_file, delimiter=":"):
        """Method for parsing simple cfguration files.
        Input: cfg_file : string : full path to input file.
        Returns dictionary of key value pairs built from each line.
        File format:
            # Comment lines begin with octothorpe.
            Data lines are <key:value>, or replace : with other delimiter.
                Value must be valid input to eval().
        Example of config file included at end of file.
        """
        to_return = {}
        if os.path.exists(cfg_file):
            with open(cfg_file, "r") as f:
                for line in f:
                    if len(line.strip()) > 0 and line.strip()[0] == "#":
                        # Skip comment lines.
                        continue
                    if line.find("#") < 0:
                        line = line[: line.find("#")]  # Allow inline comments.
                    if delimiter in line:
                        line_list = line.split(delimiter)
                        to_return[line_list[0]] = eval(line_list[1].strip())
        return to_return


class ModulatedMSExemplar(QSCOUTBuiltins, HelperFunctions):
    # This class inherits both QSCOUTBuiltins and HelperFunctions

    def gate_Mod_MS(self, channel1, channel2, singleion=False, global_duration=-1e6):
        """General Modulated MS Gate (Produce optimal pulses found by solver).
        Typically want to read in amplitude, frequency, and phase from cfg file.
        In the manual input version of this example, the global beam does a
        square pulse with constant frequency. The individual beams do a Gaussian
        pulse with symmetric frequency modulated near the red and blue motional
        sidebands."""
        ## Note, tuple input to PulseData indicates a spline. List is instant jumps.

        ## Calculate relevant values from the calibrated parameters.
        # Use calibrated, matched pi time for baseline Rabi rate.
        rabi_rate_0 = 0.5 / self.counter_resonant_pi_time
        # Use global beam as the lower leg of the Raman transition.
        global_beam_frequency = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)

        from_file = True
        if from_file:
            ## Use a cfguration file to read in waveform parameters.
            folder = os.path.dirname(os.path.abspath(__file__))
            cfg_file = os.path.join(folder, r"Exemplar_ModulatedMS_Config.txt")
            cfg = self.get_cfg(cfg_file)
            # Get Rabi rate knots and scaling from the configuration file.
            rabi_fac = cfg["rabi_fac"]
            rabi_knots = cfg["rabi_knots"]
            # Get detuning knots relative to the carrier.
            detuning_knots = cfg["detuning_knots"]
            # Get phase steps (jumps, not spline knots) and convert to degrees.
            phase_steps = [p * 180 / np.pi for p in cfg["phase_steps"]]
            if len(phase_steps) < 1:
                phase_steps = [
                    0,
                    0,
                ]  # if there are no knots, default to constant phase list.

            # is_gaussian is used in frame rotation decisions
            is_gaussian = cfg["is_gaussian"]

        else:
            ## Manually pass in waveform parameters.
            # rabi_fac is used as an overall Rabi rate scaling factor.
            rabi_fac = 1
            # rabi_knots (in Hz) are desired rabi rate knots.
            # This example is 13 points along a Gaussian with a height of the nominal Rabi rate
            rabi_knots = self.gauss(npoints=13, A=rabi_rate_0)
            # detuning_knots are desired frequency relative to the carrier.
            # This example is a sweep from 100kHz to 10kHz below the lowest frequency mode.
            detuning_knots = [
                self.lower_motional_mode_frequencies[-1] - 100e3,
                self.lower_motional_mode_frequencies[-1] - 50e3,
                self.lower_motional_mode_frequencies[-1] - 10e3,
                self.lower_motional_mode_frequencies[-1] - 50e3,
                self.lower_motional_mode_frequencies[-1] - 100e3,
            ]
            # phase_steps are the phase steps applied to the global beam.
            # This example has constant phase.
            phase_steps = [0, 0]

            # is_gaussian is used in frame rotation decisions
            is_gaussian = True

        # Convert Rabi rate knots to an amplitude scale. Default to a square pulse if no input.
        if len(rabi_knots) > 1:
            amp_scale = [rabi_fac * rk / rabi_rate_0 for rk in rabi_knots]
        else:
            amp_scale = [1, 1]
        amp_scale = np.array(amp_scale)

        # Convert detuning knots to actual RF drive frequencies. Blue=fm0, Red=fm1
        freq_fm0 = tuple(
            [
                discretize_frequency(self.ia_center_frequency)
                + discretize_frequency(self.MS_delta)
                + discretize_frequency(dk)
                for dk in detuning_knots
            ]
        )
        freq_fm1 = tuple(
            [
                discretize_frequency(self.ia_center_frequency)
                - discretize_frequency(self.MS_delta)
                - discretize_frequency(dk)
                for dk in detuning_knots
            ]
        )

        # ERF Stark shift correction for Gaussian pulses. Constant otherwise.
        if is_gaussian:
            framerot_input = tuple(
                self.erf(
                    len(amp_scale),
                    self.MS_framerot * amp_scale,
                    freqwidth=300e3,
                    total_duration=4e-6,
                )
            )
            framerot_app = 0
        else:
            framerot_input = self.MS_framerot
            framerot_app = 0b01

        # If we are scanning global_duration parameter, generate a list of
        #   amplitudes to shut off the global beam after global_duration.
        if global_duration >= 0 and global_duration < self.MS_pulse_duration:
            global_amp = [
                self.amp0_counterprop if t <= global_duration else 0
                for t in np.linspace(0, self.MS_pulse_duration, 1000)
            ]
        else:
            global_amp = self.amp0_counterprop

        listtoReturn = [
            PulseData(
                GLOBAL_BEAM,
                self.MS_pulse_duration,
                freq1=self.global_center_frequency,
                amp0=global_amp,
                phase0=phase_steps,
                phase1=0,
                sync_mask=0b11,
                fb_enable_mask=0b01,
            ),
            PulseData(
                channel1,
                self.MS_pulse_duration,
                freq0=freq_fm0,
                freq1=freq_fm1,
                amp0=tuple(self.MS_blue_amp_list[channel1] * amp_scale),
                amp1=tuple(self.MS_red_amp_list[channel1] * amp_scale),
                phase0=0,
                phase1=0,
                framerot0=framerot_input,
                apply_at_end_mask=framerot_app,
                fwd_frame0_mask=both_tones,
                sync_mask=0b11,
                fb_enable_mask=0b00,
            ),
        ]
        if not singleion:
            listtoReturn.append(
                PulseData(
                    channel2,
                    self.MS_pulse_duration,
                    freq0=freq_fm0,
                    freq1=freq_fm1,
                    amp0=tuple(self.MS_blue_amp_list[channel2] * amp_scale),
                    amp1=tuple(self.MS_red_amp_list[channel2] * amp_scale),
                    phase0=0,
                    phase1=0,
                    framerot0=framerot_input,
                    apply_at_end_mask=framerot_app,
                    fwd_frame0_mask=both_tones,
                    sync_mask=0b11,
                    fb_enable_mask=0b00,
                )
            )
        return listtoReturn


class jaqal_pulses:
    GatePulses = ModulatedMSExemplar


""" Copy of Exemplar_ModulatedMS_Config.txt below :
# Example configuration file that reproduces a Gaussian FM gate from manual input.
#   (Assumes nominal rabi rate of 100kHz and sideband frequency of 2.1 MHz.)
rabi_fac:1
rabi_knots:[82.00746635147082, 719.1883355826374, 4249.905628536256, 16922.454248245, 45404.07387272451, 82086.87174155397, 100000.0, 82086.871741554, 45404.073872724526, 16922.454248245012, 4249.905628536264, 719.1883355826387, 82.00746635147082]
detuning_knots:[2000000, 2050000, 2090000, 2050000, 2000000]
phase_steps:[0, 0]
is_gaussian:True

"""

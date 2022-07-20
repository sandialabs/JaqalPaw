################  QSCOUTBuiltins SAND Number: SAND2021-11361 O ################
""" This JaqalPaw file should be imported by user JaqalPaw code, and the
composite class, QSCOUTBuiltins, should be inherited in order to gain access
to CalibrationParameters and StandardJaqalGates. Users should only reference
members of CalibrationParameters, StandardJaqalGates, and constants at the
beginning of the file. Nothing else is gauranteed to be the same as this file.

If the structure of user accessible fields changes, the posted version of
this file will be updated at: https://qscout.sandia.gov

Last Update: September 14, 2021
"""
import abc

from jaqalpaw.utilities.datatypes import Spline, Discrete, Loop, Parallel, Sequential
from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
from jaqalpaw.utilities.exceptions import PulseException
import numpy as np
from scipy.special import erf as _erf
import logging
from math import pi
import os

## Constant Values.
# Tone mask binary values for convenience.
tone0 = 0b01
tone1 = 0b10
no_tones = 0b00
both_tones = 0b11

# RFSoC Channel Connections
GLOBAL_BEAM = 0
QN1 = 1
Q0 = 2
Q1 = 3
MICROWAVE = 4
PUMP = 5
MONITOR = 7


class CalibrationParameters:
    """Class that contains calibrated physical parameters and mapping
    information to the experimental apparatus.
    These parameters are overwritten with most recent calibrated values when
    code is run.
    Type annotations are used to mark variables to expose them to the experiment
    control software, actual type specified is not enforced."""

    ## Raman carrier transition splitting and AOM center frequencies.
    global_center_frequency: float = 200e6
    ia_center_frequency: float = 230e6
    adjusted_carrier_splitting: float = 30e6

    ## Principal axis rotation (relative to Raman k_effective).
    principal_axis_rotation: float = 45.0

    ## Motional mode frequencies.
    # Just 2 Ions in this example, list structure extends to N.
    higher_motional_mode_frequencies: list = [2.55e6, 2.45e6]
    lower_motional_mode_frequencies: list = [2.1e6, 2.05e6]

    ## Matched pi time for single qubit gates.
    co_ia_resonant_pi_time: float = 30e-6
    counter_resonant_pi_time: float = 4e-6

    ## Amplitudes to achieve matched pi times.
    # Amplitude lists are indexed by RFSoC channel. [global,-,q0,q1,-,-,-,-]
    amp0_coprop_list: list = [100, 0, 30, 30, 0, 0, 0, 30]
    amp1_coprop_list: list = [100, 0, 30, 30, 0, 0, 0, 30]
    amp0_counterprop: float = 100.0
    amp1_counterprop_list: list = [0, 0, 30, 30, 0, 0, 0, 30]

    ## Molmer Sorensen Gate Parameters
    MS_pulse_duration: float = 1e-6
    MS_delta: float = 0.0
    MS_framerot: float = 0.0
    MS_red_amp_list: list = [0, 0, 35, 30, 0, 0, 0, 35]
    MS_blue_amp_list: list = [0, 0, 33, 27, 0, 0, 0, 33]
    MS_global_amp: float = 100.0

    ## Qubit mapping.
    num_qubits: int = 2  # number of qubits used, mainly for SB cooling
    target0: int = Q0
    target1: int = Q1
    target2 = 2
    target3 = 3
    target4 = 4
    target5 = 5
    target6 = 6  # This channel is used for RF Feedback
    target7 = MONITOR

    @property
    def qubit_mapping(self):
        """This can be used to support standard declarations in Jaqal for q[0],
        q[1] etc. by mapping a whole list of targets which are currently
        controlled via the scalar variables, only some of which are exposed to
        the context window via annotations"""
        return [
            self.target0,
            self.target1,
            self.target2,
            self.target3,
            self.target4,
            self.target5,
            self.target6,
            self.target7,
        ]


class ApparatusParameters:
    """QSCOUT internal parameters, no gauranteed structure. Mostly relate to
    sideband cooling and calibration scan parameters."""

    pulse_duration: float = 1e-6
    wait_time: float = 1e-6

    microwave_frequency: float = 42e6
    microwave_pi_time: float = 100e-6

    amp0_coprop_global: float = 0.0
    amp1_coprop_global: float = 0.0

    ## Sideband Cooling Parameters
    # Red sideband pi times. Outer index refers to ion number. Inner index refers to mode.
    rsb_higher_mm_pi_times: list = [[1e-6, 1e-6], [1e-6, 1e-6]]
    rsb_lower_mm_pi_times: list = [[1e-6, 1e-6], [1e-6, 1e-6]]

    amp0_global_SBC: float = 200
    amp1_single_SBC: float = 200

    pump_time: float = 10e-6
    pump_freq: float = 222e6
    pump_amp: float = 200

    zerochannel = 2  # q0 channel
    sb_cool_gap_time: float = 1e-6
    sb_gap_time: float = 1e-6
    global_delay: float = 1e-6

    n_cool_loops: int = 10
    do_sideband_cool1: int = 0
    do_sideband_cool2: int = 0

    SK1_framerot: float = 0

    @property
    def higher_rsb_pi_times(self):
        return self.rsb_higher_mm_pi_times

    @property
    def lower_rsb_pi_times(self):
        return self.rsb_lower_mm_pi_times

    @property
    def acs0_coprop_list(self):
        return [
            self.acs0_coprop_global,
            self.acs0_coprop_qn1,
            self.acs0_coprop_q0,
            self.acs0_coprop_q1,
        ] * 2

    @property
    def acs1_coprop_list(self):
        return [
            self.acs1_coprop_global,
            self.acs1_coprop_qn1,
            self.acs1_coprop_q0,
            self.acs1_coprop_q1,
        ] * 2

    @property
    def acs_counterprop_list(self):
        return [
            self.acs0_counter_global,
            self.acs1_counter_qn1,
            self.acs1_counter_q0,
            self.acs1_counter_q1,
        ] * 2

    @property
    def phase_offset_list(self):
        return [0, 0, -100, -60] * 2


class UtilityPulses:
    def gate_measure_all(self, num_channels=8):
        return [PulseData(ch, 5e-6, freq0=0) for ch in range(num_channels)] * 2

    def wait_trigger(self, num_channels=8):
        """Sort of like the old gate_prepare_all, but not accessible as a gate and for use in macros."""
        return [
            PulseData(ch, 3e-7, rst_frame_mask=0b11, waittrig=True)
            for ch in range(num_channels)
        ]

    def gate_SBCoolHMultimode(self, *channels):
        ret = []
        max_time = 0
        for i, ch in enumerate(channels):
            max_time = max(max_time, self.higher_rsb_pi_times[ch - self.zerochannel][i])
            ret.append(
                PulseData(
                    ch,
                    self.higher_rsb_pi_times[ch - self.zerochannel][i],
                    freq1=self.global_center_frequency
                    + self.adjusted_carrier_splitting
                    - self.higher_motional_mode_frequencies[i],
                    amp1=self.amp1_single_SBC,
                    sync_mask=0b10,
                )
            )
        ret.append(
            PulseData(
                GLOBAL_BEAM,
                max_time,
                freq0=self.global_center_frequency,
                amp0=self.amp0_global_SBC,
                sync_mask=0b01,
                fb_enable_mask=0b01,
            )
        )

        return ret

    def gate_SBCoolLMultimode(self, *channels):
        ret = []
        max_time = 0
        for i, ch in enumerate(channels):
            max_time = max(max_time, self.lower_rsb_pi_times[ch - self.zerochannel][i])
            ret.append(
                PulseData(
                    ch,
                    self.lower_rsb_pi_times[ch - self.zerochannel][i],
                    freq1=self.global_center_frequency
                    + self.adjusted_carrier_splitting
                    - self.lower_motional_mode_frequencies[i],
                    amp1=self.amp1_single_SBC,
                    sync_mask=0b10,
                )
            )
        ret.append(
            PulseData(
                GLOBAL_BEAM,
                max_time,
                amp0=self.amp0_global_SBC,
                freq0=self.global_center_frequency,
                sync_mask=0b01,
                fb_enable_mask=0b01,
            )
        )
        return ret

    def gate_Pump(self):
        return [
            PulseData(PUMP, self.pump_time, freq0=self.pump_freq, amp0=self.pump_amp)
        ]

    def gate_Wait(self, channel=0, duration_scale=1):
        return [PulseData(channel, self.wait_time, freq0=0, freq1=0, enable_mask=0)]

    def gate_RCounter(self, channel, angle=np.pi, phase=0, sb_freq=0):
        duration = (angle / np.pi) * self.counter_resonant_pi_time
        global_beam_frequency = (
            discretize_frequency(self.ia_center_frequency)
            - discretize_frequency(self.adjusted_carrier_splitting)
            - discretize_frequency(sb_freq)
        )
        return [
            PulseData(
                GLOBAL_BEAM,
                duration,
                freq0=global_beam_frequency,
                amp0=self.amp0_counterprop,
                phase0=0,
                sync_mask=tone0,
                fb_enable_mask=no_tones
                if global_beam_frequency < self.ia_center_frequency
                else tone0,
            ),
            PulseData(
                channel,
                duration,
                freq1=discretize_frequency(self.ia_center_frequency),
                amp1=self.amp1_counterprop_list[int(channel)],
                phase1=phase,
                sync_mask=tone1,
                fb_enable_mask=no_tones
                if global_beam_frequency < self.ia_center_frequency
                else tone1,
                fwd_frame0_mask=tone1,
            ),
        ]

    def gate_RCoIA(self, channel, angle=np.pi, phase=0):
        lower_leg_frequency = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)
        return [
            PulseData(
                channel,
                (angle / np.pi) * self.co_ia_resonant_pi_time,
                freq0=lower_leg_frequency,
                freq1=discretize_frequency(self.ia_center_frequency),
                amp0=self.amp0_coprop_list[int(channel)],
                amp1=self.amp1_coprop_list[int(channel)],
                phase0=0,
                phase1=phase,
                sync_mask=both_tones,
                fb_enable_mask=tone0,
                fwd_frame0_mask=tone1,
            )
        ]


class DynamicalDecouplingGates:
    def gate_SK1_counter(self, channel, angle, phase1=0):
        """SK1 pulse with discrete phase updates"""
        # This is a specialized single-qubit gate and is currently only used in the phase-error correction of the MS gate
        phscorr = np.arccos(-abs(angle) / (4 * np.pi)) * 180 / np.pi
        duration = abs(2 * angle / np.pi * self.pulse_duration)
        if duration == 0:
            duration_4pi = 0
        else:
            duration_4pi = 8 * self.pulse_duration  # Turn on for SK1
            # duration_4pi = 0 #Turn on for bare gates
        duration_pi2 = self.pulse_duration
        phase = phase1 if angle > 0 else phase1 + 180
        qchannel = self.qubit_mapping[channel]
        framerot_input = self.SK1_framerot / (self.pulse_duration + duration_4pi)
        lower_freq = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)
        upper_freq = discretize_frequency(self.ia_center_frequency)
        return [
            PulseData(
                GLOBAL_BEAM,
                duration + duration_4pi,
                freq0=lower_freq,
                amp0=self.amp0_counterprop,
                sync_mask=tone0,
                fb_enable_mask=tone0,
            ),
            PulseData(
                qchannel,
                duration,
                freq1=upper_freq,
                amp1=self.amp1_counterprop_list[int(qchannel)],
                phase1=phase,
                framerot0=(0, framerot_input * duration),
                apply_at_end_mask=0,
                sync_mask=both_tones,
                fb_enable_mask=tone0,
                fwd_frame0_mask=tone1,
            ),
            PulseData(
                qchannel,
                duration_4pi,
                freq1=upper_freq,
                amp1=self.amp1_counterprop_list[int(qchannel)],
                phase1=[phase + phscorr, phase - phscorr],
                framerot0=(0, framerot_input * duration_4pi),
                apply_at_end_mask=0,
                sync_mask=both_tones,
                fb_enable_mask=tone0,
                fwd_frame0_mask=tone1,
            ),
        ]

    def gate_SK1_coprop(self, channel, angle, phase1=0):
        """SK1 pulse with discrete phase updates"""
        # This is the standard single-qubit gate used on QSCOUT, higher fidelity and not sensitive to heating
        phscorr = np.arccos(-abs(angle) / (4 * np.pi)) * 180 / np.pi
        duration = abs(2 * angle / np.pi * self.co_ia_resonant_pi_time / 2)
        if duration == 0:
            duration_4pi = 0
        else:
            duration_4pi = 8 * self.pulse_duration  # Turn on for SK1
            # duration_4pi = 0 #Turn on for bare gates
        duration_pi2 = self.co_ia_resonant_pi_time / 2
        phase = phase1 if angle > 0 else phase1 + 180
        qchannel = self.qubit_mapping[channel]
        framerot_input = self.SK1_framerot / (duration_pi2 + duration_4pi)
        lower_freq = discretize_frequency(
            self.ia_center_frequency
        ) - discretize_frequency(self.adjusted_carrier_splitting)
        upper_freq = discretize_frequency(self.ia_center_frequency)
        return [
            PulseData(
                qchannel,
                duration,
                freq0=lower_freq,
                freq1=upper_freq,
                amp0=self.amp0_coprop_list[int(qchannel)],
                amp1=self.amp1_coprop_list[int(qchannel)],
                phase1=phase,
                framerot0=(0, framerot_input * duration),
                fwd_frame0_mask=tone1,
                sync_mask=0b11,
            ),
            PulseData(
                qchannel,
                duration_4pi,
                freq0=lower_freq,
                freq1=upper_freq,
                amp0=self.amp0_coprop_list[int(qchannel)],
                amp1=self.amp1_coprop_list[int(qchannel)],
                phase1=[phase + phscorr, phase - phscorr],
                framerot0=(0, framerot_input * duration_4pi),
                fwd_frame0_mask=tone1,
                sync_mask=both_tones,
                fb_enable_mask=tone0,
            ),
        ]

    def gate_SK1(self, channel, angle, phase1=0):
        # This wrapper calls the co-propagating version of the single-qubit gate, but can be swapped to call the counter-propagating version
        return self.gate_SK1_coprop(channel, angle, phase1)


class StandardJaqalGates:
    """All single Qubit Gates as defined in Jaqal, except R

    This class extends a definition of the Jaqal R gate to
    the other standard single qubit Jaqal gates."""

    @abc.abstractmethod
    def gate_R(self, channel, angle):
        pass

    def gate_Rx(self, channel, angle):
        return self.gate_R(channel, phase=0, angle=angle)

    def gate_Ry(self, channel, angle):
        return self.gate_R(channel, phase=90, angle=angle)

    def gate_Rz(self, channel, angle):
        return [
            PulseData(
                self.qubit_mapping[channel],
                100e-9,
                framerot0=-angle * 180 / np.pi,
            )
        ]

    def gate_Px(self, channel):
        return self.gate_Rx(channel, np.pi)

    def gate_Py(self, channel):
        return self.gate_Ry(channel, np.pi)

    def gate_Pz(self, channel):
        return self.gate_Rz(channel, np.pi)

    def gate_Pxd(self, channel):
        return self.gate_Rx(channel, -np.pi)

    def gate_Pyd(self, channel):
        return self.gate_Ry(channel, -np.pi)

    def gate_Pzd(self, channel):
        return self.gate_Rz(channel, -np.pi)

    def gate_Sx(self, channel):
        return self.gate_Rx(channel, np.pi / 2)

    def gate_Sy(self, channel):
        return self.gate_Ry(channel, np.pi / 2)

    def gate_Sz(self, channel):
        return self.gate_Rz(channel, np.pi / 2)

    def gate_Sxd(self, channel):
        return self.gate_Rx(channel, -np.pi / 2)

    def gate_Syd(self, channel):
        return self.gate_Ry(channel, -np.pi / 2)

    def gate_Szd(self, channel):
        return self.gate_Rz(channel, -np.pi / 2)


class Macros:
    def macro_prepare_all(self, num_channels=8):
        gate_list = [self.wait_trigger(num_channels)]
        if self.n_cool_loops:
            loop_list = []
            if self.do_sideband_cool1:
                loop_list += [
                    self.gate_SBCoolHMultimode(
                        *[self.qubit_mapping[targ] for targ in range(self.num_qubits)]
                    ),
                    self.gate_Pump(),
                    self.gate_measure_all(),
                ]
            if self.do_sideband_cool2:
                loop_list += [
                    self.gate_SBCoolLMultimode(
                        *[self.qubit_mapping[targ] for targ in range(self.num_qubits)]
                    ),
                    self.gate_Pump(),
                    self.gate_measure_all(),
                ]
            if loop_list:
                # using the 'Loop' construct here is similar to
                # Jaqal and just offers a little compiler speedup
                gate_list += [Loop(loop_list, repeats=self.n_cool_loops)]
                # Add in 5 extra pump cycles
                gate_list += [self.gate_Pump() for i in range(5)]
                gate_list += [self.gate_measure_all()]
        return gate_list


class QSCOUTBuiltins(
    CalibrationParameters,
    ApparatusParameters,
    UtilityPulses,
    DynamicalDecouplingGates,
    StandardJaqalGates,
    Macros,
):
    pass

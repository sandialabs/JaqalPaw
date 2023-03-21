import os
from pathlib import Path
import unittest

from jaqalpaw.ir.pulse_data import PulseData
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins
from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler


class TestGatePulses(QSCOUTBuiltins):
    def gate_G(self, qubit):
        return [
            PulseData(
                qubit,
                1.25e-6,
                freq0=200e6,
                amp0=50,
                phase0=0
            )
        ]

    def gate_G0(self, qubit):
        ''' Scalar freq0 and amp0 '''
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=20)]

    def gate_G1(self, qubit):
        ''' Scalar freq0 and linear interpolated amp0 '''
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=[10, 30, 20, 50],
                          phase0=0)]

    def gate_G2(self, qubit):
        ''' Scalar freq0 and spline interpolated amp0 '''
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=(10, 30, 20, 50),
                          phase0=0)]

    def gate_G3(self, qubit):
        ''' Scalar linear interpolated freq0 and scalar amp0 '''
        return [PulseData(qubit, 5e-6, freq0=200e6, amp0=[10, 30, 20, 50],
                          phase0=0)]

    def gate_gap(self, qubit):
        return [PulseData(qubit, .25e-6)]


# class jaqal_pulses:
#     GatePulses = TestGatePulses


class TestBackwardsCompatibility(unittest.TestCase):
    def test_simple_backwards_compatibility_file(self):
        this_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        jaqal_file_path = this_dir / "backcompat.jaqal"
        cc = CircuitCompiler(file=jaqal_file_path)
        cc.compile()

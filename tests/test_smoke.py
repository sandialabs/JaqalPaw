import unittest


class SmokeTester(unittest.TestCase):
    def test_smoketest(self):
        # Simple test to confirm that we can import the modules
        import jaqalpaw.bytecode.binary_conversion
        import jaqalpaw.bytecode.encoding_parameters
        import jaqalpaw.bytecode.lut_programming
        import jaqalpaw.bytecode.pulse_binarization
        import jaqalpaw.bytecode.spline_mapping
        import jaqalpaw.compiler.jaqal_compiler
        import jaqalpaw.compiler.time_ordering
        import jaqalpaw.emulator.arbiters
        import jaqalpaw.emulator.byte_decoding
        import jaqalpaw.emulator.firmware_emulator
        import jaqalpaw.emulator.pdq_spline
        import jaqalpaw.emulator.uram
        import jaqalpaw.ir.ast_utilities
        import jaqalpaw.ir.circuit_constructor
        import jaqalpaw.ir.circuit_constructor_visitor
        import jaqalpaw.ir.gate_slice
        import jaqalpaw.ir.padding
        import jaqalpaw.ir.pulse_data
        import jaqalpaw.utilities.datatypes
        import jaqalpaw.utilities.exceptions
        import jaqalpaw.utilities.helper_functions
        import jaqalpaw.utilities.parameters

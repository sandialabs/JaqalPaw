from jaqalpaq.parser import parse_jaqal_string, parse_jaqal_file
from jaqalpaq.parser.parser import parse_jaqal_string_header, parse_jaqal_file_header
from jaqalpaq.core.algorithm import expand_macros, fill_in_let

from .circuit_constructor_visitor import convert_circuit_to_gateslices
from .pulse_data import PulseData
from .ast_utilities import get_let_constants
from jaqalpaw.utilities.exceptions import CircuitCompilerException
from jaqalpaw._import import get_jaqal_pulses

# ######################################################## #
# ------ Convert jaqal AST to GateSlice IR Layer --------- #
# ######################################################## #


class CircuitConstructor:
    """Walks the jaqal AST and constructs a list of GateSlice
    objects padding gaps with NOPs and ensuring no collisions"""

    def __init__(self, channel_num, pulse_definition):
        self.channel_num = channel_num
        self.slice_list = []
        self.pulse_definition = pulse_definition
        self.exported_constants = None
        self.reg_list = None
        self.gate_pulse_info = None
        # The circuit before any transformations have been done
        self.base_circuit = None
        # The circuit after certain transformations (such as
        # overriding let variables) has occurred.
        self.circuit = None

    def get_dependencies(self):
        if self.file is None:
            circuit, extra = parse_jaqal_string_header(
                self.code_literal, return_usepulses=False
            )
            self.gate_pulse_info = None
        else:
            circuit, extra = parse_jaqal_file_header(self.file, return_usepulses=True)
            usepulses = extra["usepulses"]
            self.gate_pulse_info = list(usepulses.keys())[0]
        return get_let_constants(circuit), self.gate_pulse_info

    def import_gate_pulses(self):
        if self.file is None and self.pulse_definition:
            return self.pulse_definition
        if self.gate_pulse_info is None:
            self.get_dependencies()
        if self.gate_pulse_info is None:
            raise CircuitCompilerException("No gate pulse file specified!")
        if self.file is None and self.pulse_definition:
            return self.pulse_definition
        self.pulse_definition = get_jaqal_pulses(
            ".".join(self.gate_pulse_info), self.file
        )
        return self.pulse_definition

    def generate_ast(self, file=None, override_dict=None):
        if self.base_circuit is None:
            if self.file is None:
                circuit = parse_jaqal_string(
                    self.code_literal, autoload_pulses=False, return_usepulses=False
                )
                self.gate_pulse_info = None
            else:
                circuit, extra = parse_jaqal_file(
                    self.file, autoload_pulses=False, return_usepulses=True
                )
                usepulses = extra["usepulses"]
                self.gate_pulse_info = list(usepulses.keys())[0]
            self.base_circuit = expand_macros(circuit)

        if override_dict is not None:
            self.circuit = fill_in_let(self.base_circuit, override_dict)
        else:
            self.circuit = self.base_circuit

        return self.circuit

    def construct_circuit(self, file, override_dict=None, pd_override_dict=None):
        """Generate full circuit from jaqal file. Circuit is in the form of
        PulseData objects."""
        ast = self.generate_ast(file, override_dict=override_dict)
        if pd_override_dict and isinstance(pd_override_dict, dict):
            for k, v in pd_override_dict.items():
                setattr(self.pulse_definition, k, v)
        self.slice_list = convert_circuit_to_gateslices(
            self.pulse_definition, ast, self.channel_num
        )

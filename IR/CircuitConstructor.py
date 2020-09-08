from pathlib import Path

from jaqalpaq.parser import parse_jaqal_string
from jaqalpaq.core.algorithm import expand_macros, fill_in_let
import runpy

from IR.CircuitConstructorVisitor import convert_circuit_to_gateslices
from IR.PulseData import PulseData
from IR.ast_utilities import get_let_constants
from utilities.exceptions import CircuitCompilerException

# ######################################################## #
# ------ Convert jaqal AST to GateSlice IR Layer --------- #
# ######################################################## #


class CircuitConstructor:
    """Walks the jaqal AST and constructs a list of GateSlice
       objects padding gaps with NOPs and ensuring no collisions"""

    def __init__(self, channel_num, pulse_definition):
        self.CHANNEL_NUM = channel_num
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
        ast = self.generate_ast()
        return get_let_constants(ast), self.gate_pulse_info

    def import_gate_pulses(self):
        if self.gate_pulse_info is None:
            self.get_dependencies()
        if self.gate_pulse_info is None:
            raise CircuitCompilerException("No gate pulse file specified!")
        gp_path = Path(self.file).parent
        jaqal_lets, self.gate_pulse_info = self.get_dependencies()
        gp_name = self.gate_pulse_info[-1]  # jaqal token returns a list of imports (split at '.'), last one is class name
        for p in self.gate_pulse_info[:-1]:
            gp_path /= p  # construct path object from usepulses call
        gp_path = gp_path.with_suffix('.py')
        if gp_path.exists():
            self.gate_pulse_file_path = str(gp_path)
        else:
            raise CircuitCompilerException(f"Can't find path {str(gp_path)}")
        pd_import = runpy.run_path(gp_path, init_globals={'PulseData': PulseData})
        self.pulse_definition = pd_import[gp_name]()
        return self.pulse_definition

    def generate_ast(self, file=None, override_dict=None):
        if self.base_circuit is None:
            if self.file is None:
                text = self.code_literal
            else:
                text = Path(self.file).read_text()
            circuit, extra = parse_jaqal_string(text, autoload_pulses=False, return_usepulses=True)
            usepulses = extra['usepulses']
            self.base_circuit = expand_macros(circuit)
            self.gate_pulse_info = list(usepulses.keys())[0]

        if override_dict is not None:
            self.circuit = fill_in_let(self.circuit, override_dict)
        else:
            self.circuit = self.base_circuit

        return self.circuit

    def construct_circuit(self, file, override_dict=None):
        """Generate full circuit from jaqal file. Circuit is in the form of
        PulseData objects."""
        ast = self.generate_ast(file, override_dict=override_dict)
        self.slice_list = convert_circuit_to_gateslices(
            self.pulse_definition, ast, self.CHANNEL_NUM)



from jaqalpaq.core.algorithm.visitor import Visitor

from .gate_slice import GateSlice
from .ast_utilities import (
    merge_slice_lists,
    is_total_gate,
    get_gate_data,
    normalize_number,
)
from jaqalpaw.utilities.datatypes import Loop
from jaqalpaw.utilities.exceptions import CircuitCompilerException


def convert_circuit_to_gateslices(pulse_definition, circuit, num_channels):
    """Convert a Circuit into a list of GateSlice objects."""
    visitor = CircuitConstructorVisitor(pulse_definition, num_channels)
    return visitor.visit(circuit)


def make_all_durations_equal(obj):
    """Calls obj.make_durations_equal if obj is a GateSlice.  If obj is a
       list or Loop, recursively descends to all GateSlice elements
       and calls make_durations_equal().  The argument is modified in
       place."""
    if isinstance(obj, (Loop, list)):
        for slc in obj:
            make_all_durations_equal(slc)
    else:
        obj.make_durations_equal()


class CircuitConstructorVisitor(Visitor):
    """Convert a Circuit into a list of GateSlice objects."""

    def __init__(self, pulse_definition, num_channels):
        super().__init__()
        self.pulse_definition = pulse_definition
        self.num_channels = num_channels

    def visit_Circuit(self, circuit):
        slice_list = self.visit(circuit.body)

        make_all_durations_equal(slice_list)

        return slice_list

    def visit_BlockStatement(self, block):
        """Return a list of GateSlice's or Loop's from this block."""
        slice_list = []
        if block.parallel:
            for stmt in block.statements:
                stmt_slices = self.visit(stmt)
                slice_list = merge_slice_lists(slice_list, stmt_slices)
        else:
            for stmt in block.statements:
                slice_list.append(self.visit(stmt))

        return slice_list

    def visit_GateStatement(self, gate):
        """Create a list of a single GateSlice representing this gate."""
        gslice = GateSlice(num_channels=self.num_channels)
        if not hasattr(self.pulse_definition, "gate_" + gate.name):
            raise CircuitCompilerException(f"Gate {gate.name} not found")
        if is_total_gate(gate.name):
            args = [self.num_channels]
            if len(gate.parameters) > 0:
                raise CircuitCompilerException(
                    f"gate {gate.name} cannot have parameters"
                )
        else:
            args = [self.visit(garg) for garg in gate.parameters.values()]
        gate_data = get_gate_data(self.pulse_definition, gate.name, args)
        if gate_data is not None:
            for pd in gate_data:
                if pd.dur > 3:
                    gslice.channel_data[pd.channel].append(pd)
        return [gslice]

    def visit_LoopStatement(self, loop):
        """Return a Loop object representing this loop."""
        slice_list = self.visit(loop.statements)
        return Loop(slice_list, repeats=loop.iterations)

    def visit_int(self, obj):
        """Integer gate arguments remain unchanged."""
        return obj

    def visit_float(self, obj):
        """Float gate arguments remain unchanged."""
        return normalize_number(obj)

    def visit_NamedQubit(self, qubit):
        """Return the index of this qubit in its register. The gate will know
        by its position that this is a qubit index and not an integer."""
        _, index = qubit.resolve_qubit()
        return index

    def visit_Constant(self, const):
        """Resolve the constant to a numeric value and return that."""
        ret = normalize_number(const.resolve_value())
        return ret

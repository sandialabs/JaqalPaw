from jaqalpaq.core.algorithm.visitor import Visitor

from .gate_slice import GateSlice
from .ast_utilities import (
    merge_slice_lists,
    is_total_gate,
    get_gate_data,
    normalize_number,
    get_macro_data,
)
from jaqalpaw.utilities.datatypes import Loop, Branch, Case, Parallel, Sequential
from jaqalpaw.utilities.exceptions import CircuitCompilerException
from ..bytecode.encoding_parameters import MINIMUM_PULSE_CLOCK_CYCLES


def is_block(gdata):
    """Checks if input is a block type synonym"""
    return isinstance(gdata, (Parallel, Sequential))


def is_loop(gdata):
    """Checks if instance is Loop type synonym"""
    return isinstance(gdata, Loop)


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


def populate_gate_slice(gate, num_channels):
    """Constructs a GateSlice with the relevant PulseData given by the associated PulseDefinition"""
    gslice = GateSlice(num_channels=num_channels)
    if gate is not None:
        for pd in gate:
            if pd.dur >= MINIMUM_PULSE_CLOCK_CYCLES:
                # Only append gate data if its duration is long enough
                # otherwise the gate is ignored, this is useful for
                # calibrations in which a gate duration is set to zero
                gslice.channel_data[pd.channel].append(pd)
    return gslice


class MacroConstructor:
    def __init__(self, channel_num):
        self.channel_num = channel_num
        self.slice_list = []

    @staticmethod
    def transform_gate_arg(arg):
        """Convert qubit registers to numbers, and return other arguments directly"""
        if hasattr(arg, "resolve_qubit"):
            _, index = arg.resolve_qubit()
            return index
        return arg

    def construct_gate(self, gate):
        """Constructs a GateSlice with the relevant PulseData given by the associated PulseDefinition"""
        return populate_gate_slice(gate, num_channels=self.channel_num)

    def construct_gate_block(self, gate_block):
        """Walk AST parallel/sequential blocks"""
        gslice = GateSlice(num_channels=self.channel_num)
        glist = []
        if isinstance(gate_block, Parallel):
            for g in gate_block:
                if is_block(g):
                    gslice.merge(self.construct_gate_block(g))
                else:
                    gslice.merge(self.construct_gate(g))
            return gslice.make_durations_equal()
        else:
            for g in gate_block:
                if is_block(g):
                    glist.append(self.construct_gate_block(g))
                elif is_loop(g):
                    glist.append(self.construct_gate_loop(g))
                elif isinstance(g, list) and isinstance(g[0], list):
                    glist.append(self.construct_gate_block(Sequential(g)))
                else:
                    glist.append(self.construct_gate(g).make_durations_equal())
        return glist

    def construct_gate_loop(self, g):
        """Walk AST for 'loop' blocks"""
        glist = Loop(repeats=g.repeats)
        new_slice = self.construct_gate_block(Sequential(g))
        glist.extend(new_slice)
        return glist

    def construct_circuit(self, circ_data, slice_list=None):
        """Generate full circuit from circ_data. Circuit is in the form of PulseData objects."""
        slice_list = slice_list or []
        for g in circ_data:
            if is_block(g):
                slice_list.append(self.construct_gate_block(g))
            elif is_loop(g):
                slice_list.append(self.construct_gate_loop(g))
            elif isinstance(g, list) and isinstance(g[0], list):
                self.construct_circuit(g, slice_list=slice_list)
            else:
                slice_list.append(self.construct_gate(g).make_durations_equal())
        return slice_list


class CircuitConstructorVisitor(Visitor):
    """Convert a Circuit into a list of GateSlice objects."""

    def __init__(self, pulse_definition, num_channels):
        super().__init__()
        self.pulse_definition = pulse_definition
        self.num_channels = num_channels
        self.macro_constructor = MacroConstructor(channel_num=self.num_channels)

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
        if not hasattr(self.pulse_definition, "gate_" + gate.name) and not hasattr(
            self.pulse_definition, "macro_" + gate.name
        ):
            raise CircuitCompilerException(f"Gate {gate.name} not found")
        if is_total_gate(gate.name):
            args = [self.num_channels]
            if len(gate.parameters) > 0:
                raise CircuitCompilerException(
                    f"gate {gate.name} cannot have parameters"
                )
        else:
            args = [self.visit(garg) for garg in gate.parameters.values()]
        if hasattr(self.pulse_definition, "macro_" + gate.name):
            macro_data = get_macro_data(self.pulse_definition, gate.name, args)
            return self.macro_constructor.construct_circuit(macro_data)
        gate_data = get_gate_data(self.pulse_definition, gate.name, args)
        return [populate_gate_slice(gate_data, self.num_channels)]

    def visit_LoopStatement(self, loop):
        """Return a Loop object representing this loop."""
        slice_list = self.visit(loop.statements)
        return Loop(slice_list, repeats=loop.iterations)

    def visit_BranchStatement(self, branch):
        """Return a list of GateSlice's or Loop's from this block."""
        slice_list = [self.visit(stmt) for stmt in branch.cases]
        return Branch(slice_list)

    def visit_CaseStatement(self, block):
        slice_list = self.visit(block.statements)
        return Case(slice_list, state=block.state)

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

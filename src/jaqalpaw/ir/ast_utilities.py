from itertools import zip_longest

from jaqalpaw.utilities.exceptions import CircuitCompilerException


def get_let_constants(ast):
    """Return a list mapping let constant names to their numeric values."""
    return {
        name: normalize_number(const.value) for name, const in ast.constants.items()
    }


def normalize_number(value):
    """Return an int if the value is an integer (regardless of whether it
    is represented as a float), or a float otherwise."""
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        if int(value) == value:
            return int(value)
        return value
    else:
        raise TypeError("Can only normalize ints and floats")


def is_total_gate(gate_name):
    """Return if this gate uses all available qubits without explicitly
    mentioning them as arguments."""
    return gate_name in ["prepare_all", "measure_all"]


def gate_pulse_exists(pulse_definition, gate_name):
    """Return whether the given pulse definition object has the given
    gate."""
    return hasattr(pulse_definition, make_gate_function_name(gate_name))


def get_gate_data(pulse_definition, gate_name, args):
    """Evaluate and return the gate data for a gate with the given
    arguments. The gate is looked up in pulse_definition then evaluated
    with args, which must be converted to numbers."""
    if not all(isinstance(arg, (int, float)) for arg in args):
        # This is a programming error that should be fixed
        raise CircuitCompilerException(f"Gate {gate_name}: Bad arg type in {args}")
    pulse_gate = getattr(pulse_definition, make_gate_function_name(gate_name))
    return pulse_gate(*args)


def get_macro_data(pulse_definition, macro_name, args):
    """Evaluate and return the gate data for a macro with the given
    arguments. The macro is looked up in pulse_definition then evaluated
    with args, which must be converted to numbers."""
    if not all(isinstance(arg, (int, float)) for arg in args):
        # This is a programming error that should be fixed
        raise CircuitCompilerException(f"Macro {macro_name}: Bad arg type in {args}")
    pulse_macro = getattr(pulse_definition, make_macro_function_name(macro_name))
    return pulse_macro(*args)


def make_gate_function_name(gate_name):
    """Make the name of the function to look up a gate by in the pulse
    definition object."""
    return f"gate_{gate_name}"


def make_macro_function_name(gate_name):
    """Make the name of the function to look up a macro by in the pulse
    definition object."""
    return f"macro_{gate_name}"


def merge_slice_lists(dst_list, src_list):
    """Take two lists of GateSlice objects and merge them. Overwrites
    dst_list."""

    return [merge_slices(dst, src) for dst, src in zip_longest(dst_list, src_list)]


def merge_slices(dst, src):
    """Merge two GateSlice objects. Handles the case where one or the
    other is None. This will always return one of the arguments,
    possibly modified, so the arguments must not be used afterwards.

    """

    if dst is None:
        assert src is not None
        return src
    elif src is None:
        assert dst is not None
        return dst
    else:
        dst.merge(src)
        return dst


def iter_gate_parameters(gate):
    """Iterate over gate parameters in order."""
    parameter_types = gate.gate_def.parameters
    for param in parameter_types:
        yield gate.parameters[param.name]

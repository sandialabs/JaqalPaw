import random
from tempfile import NamedTemporaryFile
import os
import shutil
from pathlib import Path

from jaqalpaq.core import Register
from jaqalpaq.core.circuitbuilder import build
from jaqalpaq.generator import generate_jaqal_program

# from octet.jaqalCompiler import CircuitCompiler
from benchmark.benchmark import Benchmark, run_benchmarks, profile_benchmarks

from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler


class ManyGates(Benchmark):
    """Benchmark running many gates without any additional syntax."""

    inner_number = 1
    outer_number = 10
    gate_count = 20
    circuit_depth = 5000

    def setUp(self):
        register = make_random_register()
        gate_set = []
        for _ in range(self.gate_count):
            gate_set.append(make_random_gate(register))
        gates = random.choices(gate_set, k=self.circuit_depth)
        sexpr = ("circuit", register, *gates)
        circ = build(make_hashable(sexpr))
        code = (
            "from pulse_definitions.StandardGatePulses usepulses *\n\n"
            + generate_jaqal_program(circ)
        )
        self.fd = NamedTemporaryFile(delete=False)
        # The code compiler expects the pulse definition file to be in
        # the same directory as the code, so until we change that we
        # do this workaround.
        copy_pulse_definition_file(Path(self.fd.name).parent)
        self.fd.write(code.encode())
        self.fd.close()

    def tearDown(self):
        os.unlink(self.fd.name)

    def run(self):
        cc = CircuitCompiler(self.fd.name)
        cc.bytecode(0xFF)


class NestedGates(Benchmark):
    """Benchmark gates with fairly deep nesting of blocks."""

    inner_number = 1
    outer_number = 10
    gate_count = 20
    circuit_depth = 1000

    def setUp(self):
        self.fd = None
        register = make_random_register()
        gate_set = []
        for _ in range(self.gate_count):
            gate_set.append(make_random_gate(register, use_all_qubit=False))
        remaining_depth = self.circuit_depth
        gates = []
        for i in range(self.circuit_depth):
            block, used = make_random_nested_parallel_block(
                register, remaining_depth, gate_set
            )
            gates.append(block)
            remaining_depth -= used
            if remaining_depth == 0:
                break
        sexpr = ["circuit", register, *gates]
        circ = build(make_hashable(sexpr))
        code = (
            "from pulse_definitions.StandardGatePulses usepulses *\n\n"
            + generate_jaqal_program(circ)
        )
        self.fd = NamedTemporaryFile(delete=False)
        # The code compiler expects the pulse definition file to be in
        # the same directory as the code, so until we change that we
        # do this workaround.
        copy_pulse_definition_file(Path(self.fd.name).parent)
        self.fd.write(code.encode())
        self.fd.close()

    def tearDown(self):
        if self.fd is not None:
            os.unlink(self.fd.name)

    def run(self):
        cc = CircuitCompiler(self.fd.name)
        cc.bytecode(0xFF)


##
# Helper functions
#


def copy_pulse_definition_file(dirname):
    """Find the PulseDefinitions.py file and copy it to the given directory."""
    filename = "pulse_definitions.py"
    # We're not sure where we are being run from, so search around to
    # find the pulse definitions file.
    dir_candidates = [Path("./examples"), Path("..") / "examples"]
    for dir in dir_candidates:
        if dir.exists() and filename in os.listdir(dir):
            shutil.copy(dir / filename, dirname)
            break
    else:
        # I don't know where we are
        raise ValueError(
            "Cannot find PulseDefinitions.py. Run benchmarks from the source directory."
        )


def make_random_register():
    """Create a register with some name and size of at least 3."""
    # The gates only have default timings for up to 5 qubits
    size = random.randint(3, 5)
    name = "r"
    return Register(name, size)


def make_random_nested_parallel_block(
    register, circuit_depth, gate_set, valid_qubits=None
):
    """Create a parallel block that may have sequential blocks nested
    within. Return the block and how many statements are contained
    within (not counting the block itself as a statement).

    """

    if valid_qubits is None:
        valid_qubits = QubitRange(0, register.size - 1)
    gates = []
    used = 0
    while circuit_depth > 0:
        choice = random.uniform(0, 1)
        if choice < 0.5 and valid_qubits.size > 1:
            # Create a nested block
            sub_qubits, valid_qubits = valid_qubits.halve()
            block, used = make_random_nested_sequential_block(
                register, circuit_depth, gate_set, sub_qubits
            )
            gates.append(block)
            circuit_depth -= used
        elif choice < 0.8 and valid_qubits.size > 0:
            # Create a gate
            sub_qubits, valid_qubits = valid_qubits.pop()
            gates.append(select_random_gate(gate_set, sub_qubits))
            used += 1
            circuit_depth -= 1
        else:
            # Stop adding to this block
            break

    return ["parallel_block", *gates], used


class QubitRange:
    """Represent a range of qubits. This is used to get around limitations
    in the Register class for testing purposes."""

    def __init__(self, low, high):
        """Bounds are inclusive"""
        self.low = int(low)
        self.high = int(high)
        if self.size < 0:
            raise ValueError("Negative size for QubitRange")

    def __contains__(self, index):
        return self.low <= index <= self.high

    @property
    def size(self):
        return self.high - self.low + 1

    def halve(self):
        """Return a qubit range representing the lower half and upper half of
        this range."""
        half = self.size / 2
        return (
            QubitRange(self.low, self.low + half - 1),
            QubitRange(self.low + half, self.high),
        )

    def pop(self):
        """Return a qubit range with just the first qubit and another with the
        rest."""
        return (QubitRange(self.low, self.low), QubitRange(self.low + 1, self.high))


def make_random_nested_sequential_block(
    register, circuit_depth, gate_set, valid_qubits=None
):
    """Create a parallel block that may have sequential blocks nested
    within. Return the block and how many statements are contained
    within (not counting the block itself as a statement).

    """

    if valid_qubits is None:
        valid_qubits = QubitRange(0, register.size - 1)
    gates = []
    used = 0
    while circuit_depth > 0:
        choice = random.uniform(0, 1)
        if choice < 0.5:
            # Create a nested block
            block, used = make_random_nested_parallel_block(
                register, circuit_depth, gate_set, valid_qubits
            )
            gates.append(block)
            circuit_depth -= used
        elif choice < 0.8:
            # Create a gate
            gates.append(select_random_gate(gate_set, valid_qubits))
            circuit_depth -= 1
            used += 1
        else:
            # Stop adding to this block
            break

    return ["sequential_block", *gates], used


def select_random_gate(gate_set, valid_qubits):
    """Select a gate from gate_set that doesn't use any qubits"""
    valid_gates = []
    for gate in gate_set:
        for arg in gate[2:]:
            if isinstance(arg, (list, tuple)) and arg[0] == "array_item":
                # Assume this is from the base register
                if arg[2] not in valid_qubits:
                    break
        else:
            valid_gates.append(gate)
    assert valid_gates
    return random.choice(valid_gates)


def make_random_gate(register, valid_qubits=None, use_all_qubit=True):
    """Create a gate from the legal gate set with appropriate
    arguments."""
    if valid_qubits is None:
        valid_qubits = QubitRange(0, register.size - 1)
    options = [
        make_random_gate_R_copropagating_square,
        make_random_gate_R,
        make_random_gate_Rx,
        make_random_gate_Ry,
        make_random_gate_Rz,
        make_random_gate_Px,
        make_random_gate_Py,
        make_random_gate_Pz,
        make_random_gate_Sx,
        make_random_gate_Sy,
        make_random_gate_Sz,
        make_random_gate_Sxd,
        make_random_gate_Syd,
        make_random_gate_Szd,
        make_random_gate_I,
    ]
    two_options = [
        # Empty unless we can account for the global beam.
    ]
    all_qubit_options = [
        make_random_gate_prepare_all,
        make_random_gate_measure_all,
        # These don't use all the qubits, but they do use the global
        # beam.
        make_random_gate_MS,
        make_random_gate_Sxx,
    ]
    if valid_qubits.size > 1:
        options = options + two_options
    if use_all_qubit:
        options = all_qubit_options + options
    return random.choice(options)(register, valid_qubits)


def make_random_gate_prepare_all(register, valid_qubits):
    return ["gate", "prepare_all"]


def make_random_gate_measure_all(register, valid_qubits):
    return ["gate", "measure_all"]


def make_random_gate_R_copropagating_square(register, valid_qubits):
    return [
        "gate",
        "R_copropagating_square",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
        make_random_angle(),
    ]


def make_random_gate_R(register, valid_qubits):
    return [
        "gate",
        "R",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
        make_random_angle(),
    ]


def make_random_gate_Rx(register, valid_qubits):
    return [
        "gate",
        "Rx",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
    ]


def make_random_gate_Ry(register, valid_qubits):
    return [
        "gate",
        "Ry",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
    ]


def make_random_gate_Rz(register, valid_qubits):
    return [
        "gate",
        "Rz",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
    ]


def make_random_gate_Px(register, valid_qubits):
    return ["gate", "Px", make_random_qubit(register, valid_qubits)]


def make_random_gate_Py(register, valid_qubits):
    return ["gate", "Py", make_random_qubit(register, valid_qubits)]


def make_random_gate_Pz(register, valid_qubits):
    return ["gate", "Pz", make_random_qubit(register, valid_qubits)]


def make_random_gate_Sx(register, valid_qubits):
    return ["gate", "Sx", make_random_qubit(register, valid_qubits)]


def make_random_gate_Sy(register, valid_qubits):
    return ["gate", "Sy", make_random_qubit(register, valid_qubits)]


def make_random_gate_Sz(register, valid_qubits):
    return ["gate", "Sz", make_random_qubit(register, valid_qubits)]


def make_random_gate_Sxd(register, valid_qubits):
    return ["gate", "Sxd", make_random_qubit(register, valid_qubits)]


def make_random_gate_Syd(register, valid_qubits):
    return ["gate", "Syd", make_random_qubit(register, valid_qubits)]


def make_random_gate_Szd(register, valid_qubits):
    return ["gate", "Szd", make_random_qubit(register, valid_qubits)]


def make_random_gate_I(register, valid_qubits):
    return ["gate", "I", make_random_qubit(register, valid_qubits)]


def make_random_gate_MS(register, valid_qubits):
    return [
        "gate",
        "MS",
        *make_random_qubits(register, valid_qubits, 2),
        make_random_angle(),
        make_random_angle(),
    ]


def make_random_gate_Sxx(register, valid_qubits):
    return ["gate", "Sxx", *make_random_qubits(register, valid_qubits, 2)]


def make_random_gate_GaussPLE2(register, valid_qubits):
    return [
        "gate",
        "GaussPLE2",
        make_random_qubit(register, valid_qubits),
        make_random_angle(),
    ]


def make_random_qubit(register, valid_qubits):
    assert valid_qubits.size >= 1
    return [
        "array_item",
        register.name,
        random.randint(valid_qubits.low, valid_qubits.high),
    ]


def make_random_qubits(register, valid_qubits, count):
    assert valid_qubits.size >= count
    indices = random.sample(range(valid_qubits.low, valid_qubits.high + 1), count)
    return [["array_item", register.name, idx] for idx in indices]


def make_random_angle():
    return random.uniform(0, 360)


def make_hashable(obj):
    """Descent into this object and replace any lists with tuples."""
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(v) for v in obj)
    else:
        return obj


def main():
    random.seed(1)  # Make deterministic
    benchmarks = [ManyGates, NestedGates]
    profile = False
    if profile:
        profile_benchmarks(benchmarks)
    else:
        run_benchmarks(benchmarks)


if __name__ == "__main__":
    main()

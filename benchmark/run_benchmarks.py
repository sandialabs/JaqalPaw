import timeit
import random
from tempfile import NamedTemporaryFile
import os
import shutil
from pathlib import Path
import math

from jaqalpaq.core import Register
from jaqalpaq.core.circuitbuilder import build
from jaqalpaq.generator import generate_jaqal_program

from octet.jaqalCompiler import CircuitCompiler


class Benchmark:

    # How many times to run the run method
    inner_number = 100000
    # How many times to call setUp, followed by many invocations of run
    outer_number = 10

    def __init__(self):
        self.times = None

    def setUp(self):
        """Override to set up environment."""
        pass

    def tearUp(self):
        """Override to tear down environment."""
        pass

    def run(self):
        """Override to provide the function under test."""
        pass

    def __call__(self):
        self.run()

    def start(self):
        self.times = []
        i = 0
        while i < self.outer_number:
            try:
                time = timeit.timeit(self, setup=self.setUp,
                                     number=self.inner_number)
                i += 1
                self.times.append(time)
            except Exception as exc:
                print(f"Ignored Exception: {exc}")
            finally:
                self.tearDown()

    @property
    def time(self):
        return sum(self.times) / len(self.times)

    @property
    def min_time(self):
        return min(self.times)

    @property
    def max_time(self):
        return max(self.times)

    def report(self):
        return f"{type(self).__name__}: {self.time} ({self.min_time}, {self.max_time}) s"


def run_benchmarks(benchmarks):
    for bm_cls in benchmarks:
        bm = bm_cls()
        bm.start()
        print(bm.report())


class ManyGates(Benchmark):
    """Benchmark running many gates without any additional syntax."""

    inner_number = 1
    outer_number = 10
    gate_count = 100
    circuit_depth = 5000

    def setUp(self):
        gate_set = []
        register = make_random_register()
        for _ in range(self.gate_count):
            gate_set.append(make_random_gate(register))
        gates = random.choices(gate_set, k=self.circuit_depth)
        sexpr = ['circuit', register, *gates]
        circ = build(sexpr)
        code = 'from PulseDefinitions.StandardGatePulses usepulses *\n\n' + \
            generate_jaqal_program(circ)
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
        cc.bytecode(0xff)


def copy_pulse_definition_file(dirname):
    """Find the PulseDefinitions.py file and copy it to the given directory."""
    filename = 'PulseDefinitions.py'
    if filename in os.listdir():
        # The cwd is probably the package root
        shutil.copy(filename, dirname)
    elif filename in os.listdir('..'):
        # The cwd is probably the directory with this file
        shutil.copy(Path('..') / filename, dirname)
    else:
        # I don't know where we are
        raise ValueError("Cannot find PulseDefinitions.py for hacky override")


def make_random_register():
    """Create a register with some name and size of at least 3."""
    # The gates only have default timings for up to 5 qubits
    size = random.randint(3, 5)
    name = 'r'
    return Register(name, size)


def make_random_gate(register):
    """Create a gate from the legal gate set with appropriate
    arguments."""
    options = [
        make_random_gate_prepare_all,
        make_random_gate_measure_all,
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
        make_random_gate_MS,
        make_random_gate_Sxx,
        make_random_gate_GaussPLE2,
    ]
    return random.choice(options)(register)


def make_random_gate_prepare_all(register):
    return ['gate', 'prepare_all']


def make_random_gate_measure_all(register):
    return ['gate', 'measure_all']


def make_random_gate_R_copropagating_square(register):
    return ['gate', 'R_copropagating_square', make_random_qubit(register),
            make_random_angle(), make_random_angle()]


def make_random_gate_R(register):
    return ['gate', 'R', make_random_qubit(register),
            make_random_angle(), make_random_angle()]


def make_random_gate_Rx(register):
    return ['gate', 'Rx', make_random_qubit(register), make_random_angle()]


def make_random_gate_Ry(register):
    return ['gate', 'Ry', make_random_qubit(register), make_random_angle()]


def make_random_gate_Rz(register):
    return ['gate', 'Rz', make_random_qubit(register), make_random_angle()]


def make_random_gate_Px(register):
    return ['gate', 'Px', make_random_qubit(register)]


def make_random_gate_Py(register):
    return ['gate', 'Py', make_random_qubit(register)]


def make_random_gate_Pz(register):
    return ['gate', 'Pz', make_random_qubit(register)]


def make_random_gate_Sx(register):
    return ['gate', 'Sx', make_random_qubit(register)]


def make_random_gate_Sy(register):
    return ['gate', 'Sy', make_random_qubit(register)]


def make_random_gate_Sz(register):
    return ['gate', 'Sz', make_random_qubit(register)]


def make_random_gate_Sxd(register):
    return ['gate', 'Sxd', make_random_qubit(register)]


def make_random_gate_Syd(register):
    return ['gate', 'Syd', make_random_qubit(register)]


def make_random_gate_Szd(register):
    return ['gate', 'Szd', make_random_qubit(register)]


def make_random_gate_I(register):
    return ['gate', 'I', make_random_qubit(register)]


def make_random_gate_MS(register):
    return ['gate', 'MS', *make_random_qubits(register, 2),
            make_random_angle(), make_random_angle()]


def make_random_gate_Sxx(register):
    return ['gate', 'Sxx', *make_random_qubits(register, 2)]


def make_random_gate_GaussPLE2(register):
    return ['gate', 'GaussPLE2', make_random_qubit(register), make_random_angle()]


def make_random_qubit(register):
    return ['array_item', register.name, random.randint(0, register.size - 1)]


def make_random_qubits(register, count):
    indices = random.sample(range(0, register.size), count)
    return [['array_item', register.name, idx] for idx in indices]


def make_random_angle():
    return random.uniform(0, 360)


def main():
    random.seed(1)  # Make deterministic
    run_benchmarks([ManyGates])


if __name__ == '__main__':
    main()

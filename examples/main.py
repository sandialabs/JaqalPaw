import time
from pathlib import Path

from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler
from jaqalpaw.emulator.firmware_emulator import plot_octet_emulator_output


def flatten(code):
    mtbytes = b""
    for q in code:
        for qq in q:
            mtbytes += b"".join(qq)
    return mtbytes


def emulate_jaqal_file(file):
    start = time.time()
    cc = CircuitCompiler(file=file)
    cc.compile()
    code = flatten(cc.bytecode(0xFF))
    stop = time.time()
    print(f"time: {stop-start}")
    # plot_octet_emulator_output(code)


def find_file(filename):
    """Look for the file in several reasonable places like the current
    directory, and examples directory."""

    candidate_dirs = [Path("."), Path("./examples"), Path("../examples")]
    for dir in candidate_dirs:
        if (dir / filename).exists():
            return dir / filename
    raise IOError(f"Could not find file {filename}")


if __name__ == "__main__":
    # emulate_jaqal_file("examples/test.jql")
    emulate_jaqal_file(find_file("hadamard.jql"))
    # emulate_jaqal_file("autogen_exp.jql")

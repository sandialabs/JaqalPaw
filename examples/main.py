# from octet.HWEmulator.FirmwareEmulator import trigger_events, printMetadataOutput, plotOctetEmulatorOutput
# from octet.jaqalCompiler import CircuitCompiler
from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler
from jaqalpaw.emulator.firmware_emulator import plot_octet_emulator_output
import time


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


if __name__ == "__main__":
    # emulate_jaqal_file("examples/test.jql")
    emulate_jaqal_file("hadamard.jql")
    # emulate_jaqal_file("autogen_exp.jql")

#from octet.HWEmulator.FirmwareEmulator import trigger_events, printMetadataOutput, plotOctetEmulatorOutput
#from octet.jaqalCompiler import CircuitCompiler
from compiler.jaqalCompiler import CircuitCompiler
from emulator.FirmwareEmulator import plotOctetEmulatorOutput
import time

def flatten(code):
    mtbytes = b''
    for q in code:
        for qq in q:
            mtbytes += b''.join(qq)
    return mtbytes


def emulate_jaqal_file(file):
    start = time.time()
    cc = CircuitCompiler(file=file)
    cc.compile()
    code = flatten(cc.bytecode(0xff))
    stop = time.time()
    print(f'time: {stop-start}')
    # plotOctetEmulatorOutput(code)


if __name__=='__main__':
    # emulate_jaqal_file("examples/test.jql")
    emulate_jaqal_file("hadamard.jql")
    # emulate_jaqal_file("autogen_exp.jql")

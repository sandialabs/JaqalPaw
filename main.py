from octet.HWEmulator.FirmwareEmulator import trigger_events, printMetadataOutput, plotOctetEmulatorOutput
from octet.jaqalCompiler import CircuitCompiler


def flatten(code):
    mtbytes = b''
    for q in code:
        for qq in q:
            mtbytes += b''.join(qq)
    return mtbytes


def emulate_jaqal_file(file):
    cc = CircuitCompiler(file=file)
    cc.compile()
    code = flatten(cc.bytecode(0xff))
    plotOctetEmulatorOutput(code)


if __name__=='__main__':
    emulate_jaqal_file("test.jql")

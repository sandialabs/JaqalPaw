
from octet.HWEmulator.FirmwareEmulator import trigger_events, printMetadataOutput, plotOctetEmulatorOutput
from octet.jaqalCompiler import CircuitCompiler
from PulseDefinitions import StandardGatePulses, CustomGateDefinitions

def flatten(code):
    mtbytes = b''
    for q in code:
        for qq in q:
            mtbytes += b''.join(qq)
    return mtbytes


if __name__=='__main__':
    pulseDef = CustomGateDefinitions
    cc = CircuitCompiler(file="test.jql",
                         pulse_definition=pulseDef(),
                         num_channels=8,
                         override_dict={},
                         global_delay=0)
    cc.compile()
    code = flatten(cc.bytecode(0xff))
    plotOctetEmulatorOutput(code)
    #triggers = trigger_events(code)
    #print(len(triggers[0]))


from jaqalpaw.utilities.exceptions import CircuitCompilerException, BranchOverflowException
from jaqalpaw.bytecode.encoding_parameters import (
    CHANNELS_PER_BOARD,
    DMA_MUX_LSB,
    PROGPLUT,
    PROGSLUT,
    PROGGLUT,
    PROG_MODE_LSB,
    PLUTW,
    SLUTDW,
    SLUTW,
    GLUTW,
    SLUT_BYTECNT,
    GLUT_BYTECNT,
    GSEQ_BYTECNT,
    PLUT_ADDR_LSB,
    SLUT_BYTECNT_LSB,
    GLUT_BYTECNT_LSB,
    GSEQ_BYTECNT_LSB,
    GSEQ_ENABLE_LSB,
    GPRGW,
    ANCILLA_WAIT_LSB,
    ANCILLA_CONTINUE_LSB,
    ANCILLA_COMPILER_TAG_BIT,
    PER_BOARD_CH_MASK,
    VERSION,
    VIRTUAL_ANCILLA_TAG_BIT,
    GateSequenceMode,
)

from jaqalpaw.bytecode.binary_conversion import int_to_bytes, bytes_to_int


def address_is_invalid(addr, allowed_address_bits):
    return addr & ((1 << allowed_address_bits) - 1) != addr


def generate_gate_pulse_sequence(gateList, mapping):
    """Construct series of address sequences to be
    packed linearly into sequence LUT"""
    addr_sequence = []
    for g in gateList:
        addr_sequence.append(mapping[g])
    return addr_sequence


def generate_gate_addr_range_LUT(FullGateList):
    """Create dictionaries that contain the raw data and addresses
    that are used to program the GLUT and SLUT, the IRGLUT contains
    and intermediate representation for inspection/debugging"""
    GLUT = dict()
    IRGLUT = dict()
    SLUT = dict()
    addrcntr = 0
    for n, l in enumerate(FullGateList):
        addrcntrloc = addrcntr
        for i in l:
            SLUT[addrcntrloc] = i
            addrcntrloc += 1
        IRGLUT[tuple(l)] = (addrcntr, addrcntr + len(l) - 1)
        GLUT[n] = (addrcntr, addrcntr + len(l) - 1)
        addrcntr += len(l)
    return IRGLUT, SLUT, GLUT

def channel_routing_data(ch):
    if VERSION == 2:
        return 1<<((ch%CHANNELS_PER_BOARD)&PER_BOARD_CH_MASK)<<DMA_MUX_LSB
    return (ch & PER_BOARD_CH_MASK) << DMA_MUX_LSB

def program_PLUT(lut, ch=0):
    """Generate programming data for the PLUT"""
    plut_PROG_list = []
    for data, addr in lut.items():
        if address_is_invalid(addr, PLUTW):
            return plut_PROG_list, addr
            raise CircuitCompilerException(
                f"PLUT programming error, address {addr} ({bin(addr)}) "
                f"exceeds maximum width of {PLUTW}"
            )
        intdata = bytes_to_int(data)
        intdata |= channel_routing_data(ch)
        intdata |= PROGPLUT << PROG_MODE_LSB
        intdata |= addr << PLUT_ADDR_LSB
        plut_PROG_list.append(int_to_bytes(intdata))
    return plut_PROG_list, None


def program_SLUT(lut, ch=0, offset=0):
    """Generate programming data for the SLUT"""
    slut_PROG_list = []
    current_byte = 0
    byte_count = 0
    BYTELIM = SLUT_BYTECNT
    for addr, data in lut.items():
        if address_is_invalid(addr, SLUTW):
            current_byte |= channel_routing_data(ch)
            current_byte |= PROGSLUT << PROG_MODE_LSB
            current_byte |= byte_count << SLUT_BYTECNT_LSB
            slut_PROG_list.append(int_to_bytes(current_byte))
            return slut_PROG_list, addr
        if byte_count >= BYTELIM:
            current_byte |= channel_routing_data(ch)
            current_byte |= PROGSLUT << PROG_MODE_LSB
            current_byte |= BYTELIM << SLUT_BYTECNT_LSB
            slut_PROG_list.append(int_to_bytes(current_byte))
            current_byte = 0
            byte_count = 0
        current_byte <<= SLUTW + SLUTDW
        current_byte |= (addr << SLUTDW) | (data & ((1<<SLUTDW)-1))
        byte_count += 1

    current_byte |= channel_routing_data(ch)
    current_byte |= PROGSLUT << PROG_MODE_LSB
    current_byte |= byte_count << SLUT_BYTECNT_LSB
    slut_PROG_list.append(int_to_bytes(current_byte))
    return slut_PROG_list, None


def remap_branch_address(addr):
    """If interleaved stream is used for overflow handling, it's much easier to
       use a contiguous address space for normal gates and offset branch sequence
       gates into a much higher address space. But for programming, we need to
       recast the address into the correct space for programming as done here."""
    if addr & (1<<VIRTUAL_ANCILLA_TAG_BIT):
        if addr & (1<<ANCILLA_COMPILER_TAG_BIT):
            raise BranchOverflowException("Branch gate exceeds bounds!")
        return addr ^ ((1<<VIRTUAL_ANCILLA_TAG_BIT) | (1<<ANCILLA_COMPILER_TAG_BIT))
    return addr


def program_GLUT(lut, ch=0):
    """Generate programming data for the GLUT"""
    glut_PROG_list = []
    current_byte = 0
    byte_count = 0
    BYTELIM = GLUT_BYTECNT
    for addr, data in lut.items():
        addr = remap_branch_address(addr)
        if address_is_invalid(addr, GPRGW):
            # only store valid addresses, for interleaved stream. Overflow
            # handling otherwise handled in jaqal_compiler.py, but need to
            # handle branch sequences so continue through rest of elements
            continue
        if byte_count >= BYTELIM:
            current_byte |= channel_routing_data(ch)
            current_byte |= PROGGLUT << PROG_MODE_LSB
            current_byte |= byte_count << GLUT_BYTECNT_LSB
            glut_PROG_list.append(int_to_bytes(current_byte))
            current_byte = 0
            byte_count = 0
        current_byte <<= 2 * SLUTW + GPRGW
        current_byte |= (addr << (2 * SLUTW)) | (data[1] << SLUTW) | data[0]
        byte_count += 1
    current_byte |= channel_routing_data(ch)
    current_byte |= PROGGLUT << PROG_MODE_LSB
    current_byte |= byte_count << GLUT_BYTECNT_LSB
    glut_PROG_list.append(int_to_bytes(current_byte))
    return glut_PROG_list, None


def tag_gseq_metadata(gseq, current_byte, byte_count, ch, wait_for_ancilla):
    """Gate sequence words pack a series of gate identifiers used for lookup
    in the GLUT. Because these gate ids are small, we can fit multiple into
    a single transfer. This function sets the metadata for indicating the
    number of gate ids being packed, and the sequence type:

        1) A standard gate sequence
        2) A gate sequence that depends on a new ancilla measurement
        3) A gate sequence that is a continuation of the previous ancilla result

    """
    if byte_count:
        current_byte |= wait_for_ancilla.value << ANCILLA_WAIT_LSB
        current_byte |= channel_routing_data(ch)
        current_byte |= 1 << GSEQ_ENABLE_LSB
        current_byte |= byte_count << GSEQ_BYTECNT_LSB
        gseq.append(int_to_bytes(current_byte))


def gate_sequence_bytes(glist, ch=0):
    """Generate gate sequence data that is input into the LUT module"""
    gseq = []
    current_byte = 0
    byte_count = 0
    BYTELIM = GSEQ_BYTECNT
    wait_for_ancilla = GateSequenceMode.standard
    for g in glist:
        if g & (1 << VIRTUAL_ANCILLA_TAG_BIT):
            g ^= (1 << VIRTUAL_ANCILLA_TAG_BIT) | (1 <<ANCILLA_COMPILER_TAG_BIT)
            if wait_for_ancilla == GateSequenceMode.standard:
                tag_gseq_metadata(gseq, current_byte, byte_count, ch, wait_for_ancilla)
                current_byte = 0
                byte_count = 0
                wait_for_ancilla = GateSequenceMode.start_branch
        else:
            if wait_for_ancilla != GateSequenceMode.standard:
                tag_gseq_metadata(gseq, current_byte, byte_count, ch, wait_for_ancilla)
                current_byte = 0
                byte_count = 0
            wait_for_ancilla = GateSequenceMode.standard
        if byte_count >= BYTELIM:  # or use_previous_ancilla:
            tag_gseq_metadata(gseq, current_byte, byte_count, ch, wait_for_ancilla)
            current_byte = 0
            byte_count = 0
            if wait_for_ancilla != GateSequenceMode.standard:
                if wait_for_ancilla == GateSequenceMode.start_branch:
                    wait_for_ancilla = GateSequenceMode.continue_branch
        current_byte |= (g & ((1 << GLUTW) - 1)) << (GLUTW * byte_count)
        byte_count += 1
    tag_gseq_metadata(gseq, current_byte, byte_count, ch, wait_for_ancilla)
    return gseq

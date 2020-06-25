from octet.intermediateRepresentations import CircuitCompilerException
from octet.encodingParameters import ENDIANNESS, DMA_MUX_OFFSET, PROGPLUT, PROGSLUT, PROGGLUT, PROG_MODE_OFFSET, \
                                     PLUTW, SLUTW, GLUTW, SLUT_BYTECNT, GLUT_BYTECNT, GSEQ_BYTECNT, \
                                     PLUT_BYTECNT_OFFSET, SLUT_BYTECNT_OFFSET, GLUT_BYTECNT_OFFSET, \
                                     GSEQ_BYTECNT_OFFSET, GSEQ_ENABLE_OFFSET

def bytes2int(b):
    return int.from_bytes(b, byteorder=ENDIANNESS, signed=False)


def int2bytes(d):
    return d.to_bytes(32, byteorder=ENDIANNESS, signed=False)


def address_is_invalid(addr, allowed_address_bits):
    return addr & (2**allowed_address_bits-1) != addr


def generateGatePulseSequence(gateList, mapping):
    """Construct series of address sequences to be
       packed linearly into sequence LUT"""
    addrSequence = []
    for g in gateList:
        addrSequence.append(mapping[g])
    return addrSequence


def generateGateAddrRangeLUT(FullGateList):
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
        IRGLUT[tuple(l)] = (addrcntr, addrcntr+len(l)-1)
        GLUT[n] = (addrcntr, addrcntr+len(l)-1)
        addrcntr += len(l)
    return IRGLUT, SLUT, GLUT


def programPLUT(lut,ch=0):
    """Generate programming data for the PLUT"""
    plutPROGList = []
    for data, addr in lut.items():
        if address_is_invalid(addr, PLUTW):
            raise CircuitCompilerException(f"PLUT programming error, address {addr} ({bin(addr)}) "
                                           f"exceeds maximum width of {PLUTW}")
        intdata = bytes2int(data)
        intdata |= (ch & 0b111) << DMA_MUX_OFFSET
        intdata |= PROGPLUT << PROG_MODE_OFFSET
        intdata |= addr << PLUT_BYTECNT_OFFSET
        plutPROGList.append(int2bytes(intdata))
    return plutPROGList


def programSLUT(lut, ch=0):
    """Generate programming data for the SLUT"""
    slutPROGList = []
    current_byte = 0
    byte_count = 0
    BYTELIM = SLUT_BYTECNT
    for addr, data in lut.items():
        if address_is_invalid(addr, SLUTW):
            raise CircuitCompilerException(f"MMAP LUT programming error, address {addr} ({bin(addr)}) "
                                           f"exceeds maximum width of {SLUTW}")
        if byte_count >= BYTELIM:
            current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
            current_byte |= PROGSLUT << PROG_MODE_OFFSET
            current_byte |= BYTELIM << SLUT_BYTECNT_OFFSET
            slutPROGList.append(int2bytes(current_byte))
            current_byte = 0
            byte_count = 0
        current_byte <<= SLUTW+PLUTW
        current_byte |= (addr << PLUTW) | data
        byte_count += 1
    current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
    current_byte |= PROGSLUT << PROG_MODE_OFFSET
    current_byte |= byte_count << SLUT_BYTECNT_OFFSET
    slutPROGList.append(int2bytes(current_byte))
    return slutPROGList


def programGLUT(lut, ch=0):
    """Generate programming data for the GLUT"""
    glutPROGList = []
    current_byte = 0
    byte_count = 0
    BYTELIM = GLUT_BYTECNT
    for addr, data in lut.items():
        if address_is_invalid(addr, GLUTW):
            raise CircuitCompilerException(f"GLUT programming error, address {addr} ({bin(addr)}) "
                                           f"exceeds maximum width of {GLUTW}")
        if byte_count >= BYTELIM:
            current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
            current_byte |= PROGGLUT << PROG_MODE_OFFSET
            current_byte |= byte_count << GLUT_BYTECNT_OFFSET
            glutPROGList.append(int2bytes(current_byte))
            current_byte = 0
            byte_count = 0
        current_byte <<= 2*SLUTW+GLUTW
        current_byte |= (addr << (2*SLUTW)) | (data[1] << SLUTW) | data[0]
        byte_count += 1
    current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
    current_byte |= PROGGLUT << PROG_MODE_OFFSET
    current_byte |= byte_count << GLUT_BYTECNT_OFFSET
    glutPROGList.append(int2bytes(current_byte))
    return glutPROGList


def gateSequenceBytes(glist, ch=0):
    """Generate gate sequence data that is input into the LUT module"""
    gseq = []
    current_byte = 0
    byte_count = 0
    BYTELIM = GSEQ_BYTECNT
    for g in glist:
        if byte_count >= BYTELIM:
            current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
            current_byte |= 1 << GSEQ_ENABLE_OFFSET
            current_byte |= BYTELIM << GSEQ_BYTECNT_OFFSET
            gseq.append(int2bytes(current_byte))
            current_byte = 0
            byte_count = 0
        current_byte |= g << (GLUTW*byte_count)
        byte_count += 1
    current_byte |= (ch & 0b111) << DMA_MUX_OFFSET
    current_byte |= 1 << GSEQ_ENABLE_OFFSET
    current_byte |= byte_count << GSEQ_BYTECNT_OFFSET
    gseq.append(int2bytes(current_byte))
    return gseq


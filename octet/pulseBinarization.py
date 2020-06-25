import numpy as np
from functools import reduce
from scipy.interpolate import CubicSpline
#from octet.intermediateRepresentations import Spline, Discrete
from octet.encodingParameters import MAXLEN, DMA_MUX_OFFSET_LOC, OUTPUT_EN_LSB_LOC, FRQ_FB_EN_LSB_LOC, \
                                     WAIT_TRIG_LSB_LOC, CLR_FRAME_LSB_LOC, APPLY_EOF_LSB_LOC, SYNC_FLAG_LSB_LOC, \
                                     AMPMOD0, AMPMOD1, FRQMOD0, FRQMOD1, PHSMOD0, PHSMOD1, FRMROT0, FRMROT1, \
                                     CLOCK_FREQUENCY, ENDIANNESS

from itertools import zip_longest

# ######################################################## #
# ------------------- Helper Functions ------------------- #
# ######################################################## #

class Spline(tuple):
    pass

class Discrete(tuple):
    pass


def convertToBytes(d, bytenum=5, signed=True):
    try:
        return d.to_bytes(bytenum, byteorder=ENDIANNESS, signed=signed)
    except OverflowError:
        print(f"{bytenum}")
        raise

def mapToBytes(data,bytenum=5):
    try:
        return reduce(lambda x, y: x+y, [d.to_bytes(bytenum, byteorder=ENDIANNESS, signed=True) for d in data])
    except OverflowError:
        print(f"Data: {data}\nBitLengths: {[len(bin(d))-2 for d in data]}")
        raise

def convertFreqFull(frqw):
    """Converts to full 40 bit frequency word for
       packing into 256 bit spline data"""
    convf = int(frqw/CLOCK_FREQUENCY*(2**40-1))
    return convf

def convertPhaseFull(phsw):
    """Converts to full 40 bit frequency word for
       packing into 256 bit spline data"""
    if abs(phsw)>=360.0:
        phsw %= 360.0
    if phsw >= 180:
        phsw -= 360
    elif phsw < -180:
        phsw += 360
    convf = int(phsw/360.0*(2**40-1))
    return convf

def convertAmpFull(ampw):
    convf = int(ampw/200.0*(2**16-1))
    fw1 = (convf << 23)
    return fw1

def delist(x):
    """For use in pulse command, just ensures that a single value
       or a list with a single value is simply treated as that value"""
    if isinstance(x, list):
        return x[0]
    else:
        return x

# ######################################################## #
# ------------ pdq Spline Coefficient Mapping ------------ #
# ######################################################## #

def cs_mapper_int(interp_table, nsteps=409625, shift_len=16):
    """Map spline coefficients into representation that
       can be used for accumulator based reconstruction.
       Convert the data to integers for hardware and bitshift
       higher order coefficients for better resolution"""
    new_coeffs = np.zeros(interp_table.shape)
    for i in range(interp_table.shape[1]):
        tstep = 1/nsteps if not isinstance(nsteps, list) else 1/nsteps[i]
        new_coeffs[3,i] = int(interp_table[3,i])
        new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
        new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
        new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
    return new_coeffs

def cs_mapper_int_auto_shift(interp_table, nsteps, apply_phase_mask=False):
    """Map spline coefficients into representation that
       can be used for accumulator based reconstruction.
       This variation automatically determines the optimal
       bit shift for a given set of spline coefficients"""
    new_coeffs = np.zeros(interp_table.shape)
    shift_len_list = []
    for i in range(interp_table.shape[1]):
        tstep = 1/nsteps[i]

        # initial pdq mapping, ordering is the same as the default output
        # of scipy's cubic spline coefficients, where index is inversely
        # related to the order of each coefficient.
        if apply_phase_mask:
            new_coeffs[3, i] = (((int(interp_table[3, i]) & 0xffffffffff) ^ 0x8000000000) - 0x8000000000)
        else:
            new_coeffs[3, i] = int(interp_table[3, i])
        new_coeffs[2,i] = interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3
        new_coeffs[1,i] = 2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3
        new_coeffs[0,i] = 6*interp_table[0,i]*tstep**3

        # coefficients have been mapped for PDQ interpolation, but the
        # higher order coefficients may be resolution limited. The data
        # can be bit-shifted on chip, and the shift is calculated here

        # number of significant bits for higher order terms
        sig_bits = np.log2(np.abs(new_coeffs[:-1, i])+1e-23)

        # number of shift bits is applied in multiples of N for Nth order coefficients, so
        # the overhead (or unused) bits are divided by the coefficient order. The shift is
        # applied to all non-zeroth order coefficients, so the maximum shift is determined
        # from the minimum overhead with multiples taken into account. The parameter width
        # is 40 bits, but is signed, so 39 sets the number of unsigned data bits. The shift
        # word is encoded in 5 bits, allowing for a maximum shift of 31. The data is written
        # to varying length words in fabric, where the word size is 40+16*N. However, if the
        # pulse time is long, more sensitivity is needed and the bit shift can exceed the
        # number of overhead bits when the coefficients are very small
        shift_len = int(min(min([(39-v)/(i+1) for i, v in enumerate(reversed(sig_bits))]), 31))

        # re-map coefficients with bit shift, bit shift is applied as 2**(shift*N) as opposed
        # to simply bit-shifting the coefficients as in int(val)<<(shift*N). This is done in
        # order to get more resolution on the LSBs, which accumulate over many clock cycles
        new_coeffs[2,i] = int(new_coeffs[2,i]*2**shift_len)
        new_coeffs[1,i] = int(new_coeffs[1,i]*2**(shift_len*2))
        new_coeffs[0,i] = int(new_coeffs[0,i]*2**(shift_len*3))

        # shift_len can vary for every coefficient, so a list of shift parameters is passed
        # along with the coefficients for metadata tagging
        shift_len_list.append(shift_len)

    return new_coeffs, shift_len_list

# ######################################################## #
# ------------------ PulseData Encoding ------------------ #
# ######################################################## #

def applyMetadata(bytelist, modtype=0, bypass=False, waittrig=False, shift_len=0, sync_mask=0,
                  enable_mask=0, fb_enable_mask=0, channel=0, apply_at_eof_mask=0, rst_frame_mask=0, ind=0):
    """Apply all metadata bits based on input flags, for splines, certain metadata bits are applied only
       with the first pulse such as waittrig, and rst_frame_mask"""
    num_padbytes = MAXLEN-len(bytelist)
    metadata = shift_len << (num_padbytes*8-8)
    if bypass:
        metadata |= 7 << (num_padbytes*8-11)
    channel_mux = channel << DMA_MUX_OFFSET_LOC
    metadata |= channel_mux
    output_en = 0
    fb_enable = 0
    tone_mask = 0b01 if modtype & 0b111 else 0b10
    if not (modtype & 0b11000000):  # frame_rot (z rotations)
        output_en = 1 << OUTPUT_EN_LSB_LOC if (enable_mask & tone_mask) else 0
        fb_enable = 1 << FRQ_FB_EN_LSB_LOC if (fb_enable_mask & tone_mask) else 0
    metadata |= output_en | fb_enable
    if ind == 0:
        sync = 0
        clr_frame = 0
        apply_at_eof = 0
        waitbit = 1 << WAIT_TRIG_LSB_LOC if waittrig else 0
        if modtype & 0b11000000:  # frame_rot (z rotations)
            clr_frame = 1 << CLR_FRAME_LSB_LOC if (rst_frame_mask & (modtype >> 6)) else 0
            apply_at_eof = 1 << APPLY_EOF_LSB_LOC if (apply_at_eof_mask & (modtype >> 6)) else 0
        else:  # normal parameters
            sync = 1 << SYNC_FLAG_LSB_LOC if (sync_mask & tone_mask) else 0
        metadata |= waitbit | sync | clr_frame | apply_at_eof
    modbyte = (int(np.log2(modtype)) << (num_padbytes*8-3))
    metadata |= modbyte
    padbytes = convertToBytes(metadata, bytenum=num_padbytes, signed=False)
    return bytelist + padbytes


def generateBytes(coeffs, xdata, wait, modtype=0, shift_len=0, bypass=False, waittrig=False, sync_mask=0,
                  enable_mask=0, fb_enable_mask=0, channel=0, apply_at_eof_mask=0, rst_frame_mask=0):
    """Generate binary data for contiguous, spline pulses. Used when pulse()
       is called and a parameter is specified by a typle"""
    final_data = []
    final_bytes = b''
    for n, x in enumerate(xdata):
        if modtype & (FRMROT0 | FRMROT1):
            v0 = 0
        else:
            v0 = int(coeffs[3, n])
        v1 = int(coeffs[2, n])
        v2 = int(coeffs[1, n])
        v3 = int(coeffs[0, n])
        v4 = int(wait) if not isinstance(wait, list) else int(wait[n])
        shift_bits = shift_len if not isinstance(shift_len, list) else shift_len[n]
        bytelist = mapToBytes([v0, v1, v2, v3, v4])
        fullbytes = applyMetadata(bytelist, modtype=modtype, bypass=bypass, waittrig=waittrig,
                                  shift_len=shift_bits, sync_mask=sync_mask, enable_mask=enable_mask,
                                  fb_enable_mask=fb_enable_mask, channel=channel,
                                  apply_at_eof_mask=apply_at_eof_mask, rst_frame_mask=rst_frame_mask, ind=n)
        final_bytes = final_bytes+fullbytes
        final_data.append(fullbytes)
    return final_bytes, final_data


def generatePulseBytes(coeffs, xdata, wait, modtype=0, bypass=False, waittrig=False, sync_mask=0,
                       enable_mask=0, fb_enable_mask=0, channel=0, apply_at_eof_mask=0, rst_frame_mask=0):
    """Generate binary data for contiguous, discrete pulses. Used when pulse()
       is called and a parameter is specified by a list"""
    final_data = []
    final_bytes = b''
    for n, x in enumerate(xdata):
        v0 = int(coeffs[n])
        v1 = 0
        v2 = 0
        v3 = 0
        v4 = int(wait) if not isinstance(wait, list) else int(wait[n])
        bytelist = mapToBytes([v0, v1, v2, v3, v4])
        fullbytes = applyMetadata(bytelist, modtype=modtype, bypass=bypass, waittrig=waittrig,
                                  sync_mask=sync_mask, enable_mask=enable_mask,
                                  fb_enable_mask=fb_enable_mask, channel=channel,
                                  apply_at_eof_mask=apply_at_eof_mask, rst_frame_mask=rst_frame_mask, ind=n)
        final_bytes = final_bytes+fullbytes
        final_data.append(fullbytes)
    return final_bytes, final_data


def generateSplineBytes(xs, ys, nsteps, pulse_mode=False, modtype=0, shift_len=0, bypass=False, waittrig=False,
                        sync_mask=0, enable_mask=0, fb_enable_mask=0, channel=0, apply_at_eof_mask=0, rst_frame_mask=0):
    """Generates spline coefficients, remaps them for a pdq spline and gets the corresponding byte data."""
    if pulse_mode:
        outbytes, final_byte_list = generatePulseBytes(ys,
                                                       xs[:],
                                                       nsteps,
                                                       modtype=modtype,
                                                       waittrig=waittrig,
                                                       bypass=bypass,
                                                       sync_mask=sync_mask,
                                                       enable_mask=enable_mask,
                                                       fb_enable_mask=fb_enable_mask,
                                                       apply_at_eof_mask=apply_at_eof_mask,
                                                       rst_frame_mask=rst_frame_mask,
                                                       channel=channel)
    else:
        cs = CubicSpline(xs, ys, bc_type=((2, 0.0), (2, 0.0)))  # set for a natural spline
        if modtype in (PHSMOD0, PHSMOD1, FRMROT0, FRMROT1):
            apply_phase_mask = True
        else:
            apply_phase_mask = False
        if shift_len < 0:
            modified_coeff_table, shift_len_fin = cs_mapper_int_auto_shift(cs.c, nsteps=nsteps,
                                                                           apply_phase_mask=apply_phase_mask)
        else:
            shift_len_fin = shift_len
            modified_coeff_table = cs_mapper_int(cs.c, nsteps=nsteps, shift_len=shift_len)
        outbytes, final_byte_list = generateBytes(modified_coeff_table,
                                                  xs[1:],
                                                  nsteps,
                                                  modtype=modtype,
                                                  shift_len=shift_len_fin,
                                                  waittrig=waittrig,
                                                  bypass=bypass,
                                                  sync_mask=sync_mask,
                                                  enable_mask=enable_mask,
                                                  fb_enable_mask=fb_enable_mask,
                                                  apply_at_eof_mask=apply_at_eof_mask,
                                                  rst_frame_mask=rst_frame_mask,
                                                  channel=channel)
    return final_byte_list


def generateSinglePulseBytes(coeff, wait, modtype=0, waittrig=False, bypass=False, sync_mask=0,
                             enable_mask=0, fb_enable_mask=0, channel=0, apply_at_eof_mask=0, rst_frame_mask=0):
    """prints out coefficient data with wait times in between.
       pow gives an overall scale factor to the data of 2**pow,
       and converts the coefficients to integers"""
    bytelist = mapToBytes([int(coeff), 0, 0, 0, int(wait)])  # v0, v1, v2, v3, duration
    final_bytes = applyMetadata(bytelist, modtype=modtype, bypass=bypass, waittrig=waittrig, sync_mask=sync_mask,
                                enable_mask=enable_mask, fb_enable_mask=fb_enable_mask, channel=channel,
                                apply_at_eof_mask=apply_at_eof_mask, rst_frame_mask=rst_frame_mask)
    return final_bytes, [final_bytes]


def pulse(DDS, dur, freq0=0, phase0=0, amp0=0, freq1=0, phase1=0, amp1=0, waittrig=False, sync_mask=0, enable_mask=0,
          fb_enable_mask=0, framerot0=0, framerot1=0, apply_at_eof_mask=0, rst_frame_mask=0, bypass=False):
    """Generates the binary data that needs to be uploaded to the chip from a set of input parameters"""
    DDS &= 0b111
    mpdict = {'freq0':     {'modtype': FRQMOD0, 'data': freq0,     'convertFunc': convertFreqFull,  'enabled': True},
              'freq1':     {'modtype': FRQMOD1, 'data': freq1,     'convertFunc': convertFreqFull,  'enabled': True},
              'amp0':      {'modtype': AMPMOD0, 'data': amp0,      'convertFunc': convertAmpFull,   'enabled': True},
              'amp1':      {'modtype': AMPMOD1, 'data': amp1,      'convertFunc': convertAmpFull,   'enabled': True},
              'phase0':    {'modtype': PHSMOD0, 'data': phase0,    'convertFunc': convertPhaseFull, 'enabled': True},
              'phase1':    {'modtype': PHSMOD1, 'data': phase1,    'convertFunc': convertPhaseFull, 'enabled': True},
              'framerot0': {'modtype': FRMROT0, 'data': framerot0, 'convertFunc': convertPhaseFull, 'enabled': True},
              'framerot1': {'modtype': FRMROT1, 'data': framerot1, 'convertFunc': convertPhaseFull, 'enabled': True}
              }
    modlist = ['freq0', 'amp0', 'phase0', 'freq1', 'amp1', 'phase1', 'framerot0', 'framerot1']
    bytelist = []
    for modt in modlist:
        if not mpdict[modt]['enabled']:
            continue
        n_points = 1 if not hasattr(mpdict[modt]['data'], '__iter__') else len(mpdict[modt]['data'])
        if n_points > 1:
            pulsemode = isinstance(mpdict[modt]['data'], Discrete)

            # xdata needs to have unit spacing to work well with the
            # pdq spline mapping even when data is nonuniform
            xdata = [i for i in range(n_points)]
            ydata = list(map(mpdict[modt]['convertFunc'], mpdict[modt]['data']))

            # raw_cycles specifies the actual time grid, distributing
            # rounding errors must have one more point for pulse mode
            # step_list is the time per point on the non-uniform grid
            raw_cycles = np.round(np.linspace(0, dur, n_points+1*pulsemode))
            step_list = list(np.diff(raw_cycles))
            if min(step_list) < 4:
                raise Exception("Step size needs to be at least 4 clock cycles, or 10 ns!")

            bytelist.append(generateSplineBytes(np.array(xdata),
                                                np.array(ydata),
                                                step_list,
                                                pulse_mode=pulsemode,
                                                modtype=mpdict[modt]['modtype'],
                                                shift_len=-1,
                                                waittrig=waittrig,
                                                bypass=bypass,
                                                sync_mask=sync_mask,
                                                enable_mask=enable_mask,
                                                fb_enable_mask=fb_enable_mask,
                                                apply_at_eof_mask=apply_at_eof_mask,
                                                rst_frame_mask=rst_frame_mask,
                                                channel=DDS))
        else:
            lbytes, _ = generateSinglePulseBytes(mpdict[modt]['convertFunc'](delist(mpdict[modt]['data'])),
                                                 dur,
                                                 modtype=mpdict[modt]['modtype'],
                                                 waittrig=waittrig,
                                                 bypass=bypass,
                                                 sync_mask=sync_mask,
                                                 enable_mask=enable_mask,
                                                 fb_enable_mask=fb_enable_mask,
                                                 apply_at_eof_mask=apply_at_eof_mask,
                                                 rst_frame_mask=rst_frame_mask,
                                                 channel=DDS)
            bytelist.append([lbytes])
    splist = list(filter(len, [m for i in zip_longest(*sorted(bytelist, key=len), fillvalue=b'') for m in i]))
    return splist




import numpy as np
from scipy.interpolate import CubicSpline
from bytecode.binaryConversion import convertFreqFull, convertAmpFull, convertPhaseFull, convertToBytes, mapToBytes
from bytecode.encodingParameters import MAXLEN, DMA_MUX_OFFSET_LOC, OUTPUT_EN_LSB_LOC, FRQ_FB_EN_LSB_LOC, \
                                     WAIT_TRIG_LSB_LOC, CLR_FRAME_LSB_LOC, APPLY_EOF_LSB_LOC, SYNC_FLAG_LSB_LOC, \
                                     AMPMOD0, AMPMOD1, FRQMOD0, FRQMOD1, PHSMOD0, PHSMOD1, FRMROT0, FRMROT1

from utilities.helper_functions import delist
from utilities.datatypes import Discrete

from itertools import zip_longest

from bytecode.spline_mapping import cs_mapper_int, cs_mapper_int_auto_shift


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


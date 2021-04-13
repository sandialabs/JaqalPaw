from collections import defaultdict
import numpy as np
from jaqalpaw.bytecode.binary_conversion import (
    convert_freq_full,
    convert_phase_full,
    convert_amp_full,
    map_from_bytes,
)
from jaqalpaw.bytecode.encoding_parameters import (
    PLUT_ADDR_LSB,
    DMA_MUX_LSB,
    GSEQ_BYTECNT_LSB,
    MODTYPE_LSB,
    SPLSHIFT_LSB,
    PROG_MODE_LSB,
    WAIT_TRIG_LSB,
    OUTPUT_EN_LSB,
    GLUT_BYTECNT_LSB,
    SLUT_BYTECNT_LSB,
)
from .uram import GLUT, SLUT, PLUT
from .pdq_spline import pdq_spline
from jaqalpaw.utilities.parameters import CLKPERIOD, CLOCK_FREQUENCY, MAXAMP
from jaqalpaw.bytecode.encoding_parameters import GPRGW, GLUTW, SLUTW, PLUTW

tree = lambda: defaultdict(tree)

mdr = tree()


def convert_phase_bytes_to_real(data):
    return data / (1 << 40) * 360.0


def convert_freq_bytes_to_real(data):
    return data / (1 << 40) * CLOCK_FREQUENCY


def convert_amp_bytes_to_real(data):
    return (int(data) >> 23) / ((1 << 16) - 1) * MAXAMP


def convert_time_from_clock_cycles(data):
    return data * CLKPERIOD


mode_enum = {
    "run": 0,
    "bypass": 1,
    "prog_plut": 2,
    "prog_slut": 3,
    "prog_glut": 4,
    None: None,
}
mode_lut = {v: k for k, v in mode_enum.items()}
mod_type_dict = {
    0b000: {
        "name": "f0",
        "machineConvFunc": convert_freq_full,
        "realConvFunc": convert_freq_bytes_to_real,
    },
    0b001: {
        "name": "a0",
        "machineConvFunc": convert_amp_full,
        "realConvFunc": convert_amp_bytes_to_real,
    },
    0b010: {
        "name": "p0",
        "machineConvFunc": convert_phase_full,
        "realConvFunc": convert_phase_bytes_to_real,
    },
    0b011: {
        "name": "f1",
        "machineConvFunc": convert_freq_full,
        "realConvFunc": convert_freq_bytes_to_real,
    },
    0b100: {
        "name": "a1",
        "machineConvFunc": convert_amp_full,
        "realConvFunc": convert_amp_bytes_to_real,
    },
    0b101: {
        "name": "p1",
        "machineConvFunc": convert_phase_full,
        "realConvFunc": convert_phase_bytes_to_real,
    },
    0b110: {
        "name": "z0",
        "machineConvFunc": convert_phase_full,
        "realConvFunc": convert_phase_bytes_to_real,
    },
    0b111: {
        "name": "z1",
        "machineConvFunc": convert_phase_full,
        "realConvFunc": convert_phase_bytes_to_real,
    },
}


def parse_GLUT_prog_data(data):
    """Program GLUT with input data word"""
    nwords = (data >> GLUT_BYTECNT_LSB) & 0b11111
    channel = (data >> (DMA_MUX_LSB)) & 0b111
    for w in range(nwords):
        sdata = data >> (w * (2 * SLUTW + GPRGW))
        glut_data = sdata & ((1 << (2 * SLUTW)) - 1)
        glut_addr = (sdata >> (2 * SLUTW)) & ((1 << GPRGW) - 1)
        GLUT[channel][glut_addr] = glut_data


def parse_SLUT_prog_data(data):
    """Program SLUT with input data word"""
    nwords = (data >> SLUT_BYTECNT_LSB) & 0b11111
    channel = (data >> DMA_MUX_LSB) & 0b111
    for w in range(nwords):
        sdata = data >> (w * (PLUTW + SLUTW))
        slut_data = sdata & ((1 << PLUTW) - 1)
        slut_addr = (sdata >> PLUTW) & ((1 << SLUTW) - 1)
        SLUT[channel][slut_addr] = slut_data


def parse_PLUT_prog_data(data):
    """Program PLUT with input data word"""
    newdata = int.from_bytes(data, byteorder="little", signed=False)
    plut_addr = (newdata >> PLUT_ADDR_LSB) & ((1 << PLUTW) - 1)
    channel = (newdata >> DMA_MUX_LSB) & 0b111
    PLUT[channel][plut_addr] = data


def iterate_GLUT_bounds(gid, channel):
    """Get all PLUT data for an individual gate"""
    bounds_bytes = GLUT[channel][gid]
    start = bounds_bytes & ((1 << SLUTW) - 1)
    stop = (bounds_bytes >> SLUTW) & ((1 << SLUTW) - 1)
    for sid in range(start, stop + 1):
        yield PLUT[channel][SLUT[channel][sid]]


def parse_gate_seq_data(data, oraddr=0):
    """Get sequence of gates to run from input data"""
    prog_byte_cnt = (data >> GSEQ_BYTECNT_LSB) & 0b111111
    channel = (data >> DMA_MUX_LSB) & 0b111
    newdata = data
    plut_list = []
    gidlist = []
    for g in range(prog_byte_cnt):
        gid = newdata & ((1 << GLUTW) - 1)
        newdata >>= GLUTW
        gidlist.append(gid | oraddr)
        for plut_data in iterate_GLUT_bounds(gid | oraddr, channel):
            plut_list.append(plut_data)
    print(f"gid list {channel}: {gidlist}")
    return plut_list


def parse_bypass_data(data):
    """Return parameters for a raw data word"""
    U0, U1, U2, U3, dur = map_from_bytes(data)
    return dur, U0, U1, U2, U3


def decode_word(raw_data, master_data_record, sequence_mode=False):
    """This function essentially acts like the data path from DMA to the spline engine output.
    Input words are 256 bits, and are parsed and treated accordingly depending on the
    metadata tags in the raw data in order to program LUTs or run gate sequences etc...
    The output is stored in a recursive default dict which is passed in to master_data_record"""
    data = int.from_bytes(raw_data, byteorder="little", signed=False)
    mod_type = (data >> MODTYPE_LSB) & 0b111
    shift = (data >> SPLSHIFT_LSB) & 0b11111
    prog_mode = (data >> PROG_MODE_LSB) & 0b111
    prog_byte_cnt = None
    channel = (data >> DMA_MUX_LSB) & 0b111
    dur, U0, U1, U2, U3 = None, None, None, None, None
    waittrig = (data >> WAIT_TRIG_LSB) & 0b1
    enablemask = (data >> OUTPUT_EN_LSB) & 0b11
    mode = None
    if prog_mode == 0b111 or sequence_mode:
        mode = mode_enum["bypass"]
        dur, U0, U1, U2, U3 = parse_bypass_data(raw_data)
    elif prog_mode == 0b001:
        prog_byte_cnt = (data >> GLUT_BYTECNT_LSB) & 0b11111111
        mode = mode_enum["prog_glut"]
        parse_GLUT_prog_data(data)
    elif prog_mode == 0b010:
        prog_byte_cnt = (data >> SLUT_BYTECNT_LSB) & 0b11111111
        mode = mode_enum["prog_slut"]
        parse_SLUT_prog_data(data)
    elif prog_mode == 0b011:
        prog_byte_cnt = (data >> PLUT_ADDR_LSB) & 0b11111111
        mode = mode_enum["prog_plut"]
        parse_PLUT_prog_data(raw_data)
    elif prog_mode == 0b100 or prog_mode == 0b101 or prog_mode == 0b110:
        prog_byte_cnt = (data >> GSEQ_BYTECNT_LSB) & 0b11111111
        mode = mode_enum["run"]
        for gs_data in parse_gate_seq_data(data):
            master_data_record = decode_word(
                gs_data, master_data_record, sequence_mode=True
            )

    print(
        f"channel: {channel}, mod type: {mod_type_dict[mod_type]['name']}, mode: {mode_lut[mode]}, shift: {shift}, prog byte count: {prog_byte_cnt}"
    )

    if mode == mode_enum["bypass"]:
        dur_real = convert_time_from_clock_cycles(dur)
        U0_real = mod_type_dict[mod_type]["realConvFunc"](U0)
        U1_real = mod_type_dict[mod_type]["realConvFunc"](U1)
        U2_real = mod_type_dict[mod_type]["realConvFunc"](U2)
        U3_real = mod_type_dict[mod_type]["realConvFunc"](U3)
        print(
            f"Duration: {dur_real} s, U0: {U0_real}, U1: {U1_real}, U2: {U2_real}, U3: {U3_real}"
        )
        if isinstance(master_data_record[channel][mod_type]["time"], defaultdict):
            master_data_record[channel][mod_type]["time"] = [0]
        if isinstance(master_data_record[channel][mod_type]["data"], defaultdict):
            master_data_record[channel][mod_type]["data"] = [0]
        if isinstance(master_data_record[channel][mod_type]["waittrig"], defaultdict):
            master_data_record[channel][mod_type]["waittrig"] = [waittrig]
        if isinstance(master_data_record[channel][mod_type]["enablemask"], defaultdict):
            master_data_record[channel][mod_type]["enablemask"] = [enablemask]
        if U1 == 0 and U2 == 0 and U3 == 0 and False:
            master_data_record[channel][mod_type]["time"].append(
                master_data_record[channel][mod_type]["time"][-1] + dur_real
            )
            master_data_record[channel][mod_type]["data"].append(U0_real)
            master_data_record[channel][mod_type]["waittrig"].append(waittrig)
            master_data_record[channel][mod_type]["enablemask"].append(enablemask)
        else:
            U1_shift = U1 / (1 << (shift * 1))
            U2_shift = U2 / (1 << (shift * 2))
            U3_shift = U3 / (1 << (shift * 3))
            U1_rshift = U1_real / (1 << shift)
            U2_rshift = U2_real / (1 << (shift * 2))
            U3_rshift = U3_real / (1 << (shift * 3))
            coeffs = np.zeros((4, 1))
            coeffs[0, 0] = U3_shift
            coeffs[1, 0] = U2_shift
            coeffs[2, 0] = U1_shift
            coeffs[3, 0] = U0
            xdata = np.array(list(range(dur))) + 1
            spline_data = pdq_spline(coeffs, [0], nsteps=dur)
            spline_data_real = list(
                map(mod_type_dict[mod_type]["realConvFunc"], spline_data)
            )
            xdata_real = list(
                map(
                    lambda x: master_data_record[channel][mod_type]["time"][-1]
                    + convert_time_from_clock_cycles(x),
                    xdata,
                )
            )
            master_data_record[channel][mod_type]["time"].extend(xdata_real)
            del master_data_record[channel][mod_type]["data"][-1]
            master_data_record[channel][mod_type]["data"].extend(spline_data_real)
            master_data_record[channel][mod_type]["data"].append(spline_data_real[-1])

            master_data_record[channel][mod_type]["waittrig"].append(
                [waittrig] + [0] * (len(xdata_real) - 1)
            )
            master_data_record[channel][mod_type]["enablemask"].append(
                [enablemask] * len(xdata_real)
            )
            print(
                f"Duration: {dur_real} s, U0: {U0}, U1: {U1_rshift}, U2: {U2_rshift}, U3: {U3_rshift}"
            )

    return master_data_record

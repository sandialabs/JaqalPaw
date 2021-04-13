from functools import reduce

from .encoding_parameters import ENDIANNESS
from jaqalpaw.utilities.parameters import CLOCK_FREQUENCY, MAXAMP


def convert_to_bytes(d, bytenum=5, signed=True):
    try:
        return d.to_bytes(bytenum, byteorder=ENDIANNESS, signed=signed)
    except OverflowError:
        print(f"{bytenum}")
        raise


def map_to_bytes(data, bytenum=5):
    try:
        return reduce(
            lambda x, y: x + y,
            [d.to_bytes(bytenum, byteorder=ENDIANNESS, signed=True) for d in data],
        )
    except OverflowError:
        print(f"Data: {data}\nBitLengths: {[len(bin(d))-2 for d in data]}")
        raise


def map_from_bytes(d, bytenum=5):
    return [
        int.from_bytes(
            d[n * bytenum : n * bytenum + bytenum], byteorder=ENDIANNESS, signed=True
        )
        for n in range(bytenum)
    ]


def bytes_to_int(b):
    return int.from_bytes(b, byteorder=ENDIANNESS, signed=False)


def int_to_bytes(d):
    return d.to_bytes(32, byteorder=ENDIANNESS, signed=False)


def signed_n_bit_map(x, n=40):
    """Convert integer x to a signed n-bit number"""
    return ((x & ~(-1 << n)) ^ (1 << n - 1)) - (1 << n - 1)


def convert_freq_full(frqw):
    """Converts to full 40 bit frequency word for
    packing into 256 bit spline data"""
    convf = int(round(frqw / CLOCK_FREQUENCY * (1 << 40)))
    return convf


def convert_phase_full(phsw):
    """Converts to full 40 bit frequency word for
    packing into 256 bit spline data"""
    convf = int(round(phsw / 360.0 * (1 << 40)))
    return convf


def convert_phase_full_mod_2pi(phsw):
    """Converts to full 40 bit frequency word for
    packing into 256 bit spline data"""
    if abs(phsw) >= 360.0:
        phsw %= 360.0
    if phsw >= 180:
        phsw -= 360
    elif phsw < -180:
        phsw += 360
    convf = int(round(phsw / 360.0 * (1 << 40)))
    return convf


def convert_amp_full(ampw):
    convf = int(ampw / MAXAMP * ((1 << 16) - 1))
    fw1 = convf << 23
    return fw1

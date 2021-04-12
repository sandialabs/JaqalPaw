from collections import UserDict, deque
from jaqalpaw.bytecode.encoding_parameters import (
    GLUTW,
    GPRGW,
    SLUTW,
    PLUTW,
    ANCILLA_COMPILER_TAG_BIT,
)


class URAMException(Exception):
    pass


class URAM(UserDict):
    """URAM mimics the Ultra RAM blocks that hold the LUTs in the gate sequencer.
    It is used like a standard simple dual port memory (one write port, one read port).
    As it functions very much like a dictionary, the URAM class is derived from UserDict,
    but adds some additional exceptions when the key/value pairs are of incorrect type or
    exceed the allowed address/data width which is allocated in the firmware."""

    def __init__(self, initialdata=None, address_width=12, data_width=10):
        super().__init__(initialdata or [])
        self.address_width = address_width
        self.data_width = data_width

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise URAMException(f"URAM address must be type int, not {type(key)}")
        if not isinstance(value, int) and not isinstance(value, bytes):
            raise URAMException(f"URAM value must be type int, not {type(value)}")
        newkey = key & ((1 << self.address_width) - 1)
        newvalue = value
        if newkey != key:
            raise URAMException(
                f"Invalid URAM address {key}, "
                "must be {self.address_width} bits and less than {2**self.address_width}"
            )
        if newvalue != value:
            raise URAMException(
                f"Invalid URAM data {value}, "
                "must be {self.data_width} bits and less than {2**self.data_width}"
            )
        self.data[key] = value


GLUT = [URAM(address_width=GPRGW, data_width=2 * SLUTW) for _ in range(8)]
SLUT = [URAM(address_width=SLUTW, data_width=PLUTW) for _ in range(8)]
PLUT = [URAM(address_width=PLUTW, data_width=256) for _ in range(8)]

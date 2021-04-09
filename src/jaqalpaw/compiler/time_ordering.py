from collections import defaultdict

from jaqalpaw.bytecode.binary_conversion import map_from_bytes
from jaqalpaw.bytecode.encoding_parameters import MODTYPE_LSB, DMA_MUX_LSB

# ######################################################## #
# --------------- Time Ordering Functions ---------------- #
# ######################################################## #


class TimeStampedWord:
    def __init__(self, duration, word, start_time=0, mod_type=None, chan=None):
        self.duration = duration
        self.word = word
        self.start_time = start_time
        self.mod_type = mod_type
        self.chan = chan

    @property
    def end_time(self):
        return self.duration + self.start_time

    def __add__(self, other):
        if not isinstance(other, TimeStampedWord):
            raise Exception(f"Can't add to object of type {type(other)}")
        elif self.chan != other.chan:
            raise Exception(f"Can't add time to other channel's data")
        else:
            return TimeStampedWord(self.duration, self.word, start_time=other.end_time)

    def __repr__(self):
        return f"type: {self.mod_type} start: {self.start_time}"


def decode_word(word):
    """Extract the channel, modulation type and duration from a data word"""
    data = int.from_bytes(word, byteorder="little", signed=False)
    mod_type = (data >> MODTYPE_LSB) & 0b111
    channel = (data >> DMA_MUX_LSB) & 0b111
    U0, U1, U2, U3, dur = map_from_bytes(word)
    return channel, mod_type, dur


def generate_time_stamped_data(bytelist):
    """Convert a list of 256 bit words to TimeStampedWord objects that
    calculates the start time for each word in the sequence"""
    parameter_dd = defaultdict(lambda: defaultdict(list))
    full_pb_list = []
    for pb in bytelist:
        chan, mod_type, dur = decode_word(pb)
        start_time = 0
        if len(parameter_dd[chan][mod_type]):
            start_time = parameter_dd[chan][mod_type][-1].end_time
        parameter_dd[chan][mod_type].append(
            TimeStampedWord(
                dur, pb, start_time=start_time, mod_type=mod_type, chan=chan
            )
        )
        full_pb_list.append(
            TimeStampedWord(
                dur, pb, start_time=start_time, mod_type=mod_type, chan=chan
            )
        )
    return full_pb_list


def timesort_bytelist(bytelist):
    """Sort a list of raw data words by start time, then by channel and modulation type"""
    full_pb_list = generate_time_stamped_data(bytelist)
    # sorted_pb_list = list(sorted(full_pb_list, key=lambda el: (el.start_time << 6) | (el.chan << 3) | el.mod_type))
    sorted_pb_list = list(
        sorted(
            full_pb_list,
            key=lambda el: (el.start_time << 6) | (el.mod_type << 3) | el.chan,
        )
    )
    wordlist = []
    for spb in sorted_pb_list:
        wordlist.append(spb.word)
    return wordlist

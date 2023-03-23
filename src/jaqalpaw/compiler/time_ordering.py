from collections import defaultdict
from functools import partial

from jaqalpaw.bytecode.binary_conversion import bytes_to_int, map_from_bytes, int_to_bytes
from jaqalpaw.bytecode.encoding_parameters import MODTYPE_LSB, DMA_MUX_LSB, VERSION
from jaqalpaw.emulator.byte_decoding import get_channels, get_mod_types

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
    mod_types = get_mod_types(data)
    channels = get_channels(data)
    U0, U1, U2, U3, dur = map_from_bytes(word)
    return channels, mod_types, dur


def generate_time_stamped_data(bytelist):
    """Convert a list of 256 bit words to TimeStampedWord objects that
    calculates the start time for each word in the sequence"""
    parameter_dd = defaultdict(lambda: defaultdict(list))
    full_pb_list = []
    for pb in bytelist:
        chans, mod_types, dur = decode_word(pb)
        start_time = 0
        for chan in chans:
            for mod_type in mod_types:
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
    sorted_pb_list = list(
        sorted(
            full_pb_list,
            key=lambda el: (el.start_time << 6) | (el.mod_type << 3) | el.chan,
        )
    )
    if VERSION == 2:
        return condense_for_broadcasting(sorted_pb_list)
    wordlist = []
    for spb in sorted_pb_list:
        wordlist.append(spb.word)
    return wordlist


routemask = (((1<<8)-1)<<DMA_MUX_LSB)| (((1<<8)-1)<<MODTYPE_LSB)

def unrouted_data(data, *, convert_to_int=False):
    convert_to_bytes = False
    if isinstance(data, bytes):
        convert_to_bytes = False if convert_to_int else True
        data = bytes_to_int(data)
    data &= ~routemask
    if convert_to_bytes:
        return int_to_bytes(data)
    return data


def separate_routing_from_data(data):
    if isinstance(data, bytes):
        data = bytes_to_int(data)
    base_data = unrouted_data(data)
    mod_types = get_mod_types(data)
    channels = get_channels(data)
    return channels, mod_types, base_data


def combine_data(common_time_data):
    separated_data = list(map(separate_routing_from_data, common_time_data))
    dd = defaultdict(lambda: defaultdict(list))
    for chs,mt,dat in separated_data:
        for ch in chs:
            dd[ch][dat].extend(mt)
    du = defaultdict(list)
    for ch,cdict in dd.items():
        for dat,mts in cdict.items():
            newdat = dat
            for mt in mts:
                newdat |= 1<<mt<<MODTYPE_LSB
            du[newdat].append(ch)
    comblist = []
    for dat,chs in du.items():
        newdat = dat & ((1<<DMA_MUX_LSB)-1)
        for ch in chs:
            newdat |= 1<<ch<<DMA_MUX_LSB
        comblist.append(int_to_bytes(newdat))
    return comblist


def condense_for_broadcasting(pblist):
    current_start_time = 0
    commont_data = []
    wordlist = []
    for pb in pblist:
        if pb.start_time == current_start_time:
            commont_data.append(pb.word)
        else:
            if commont_data:
                wordlist.extend(combine_data(commont_data))
                commont_data.clear()
            current_start_time = pb.start_time
            commont_data.append(pb.word)
    if commont_data:
        wordlist.extend(combine_data(commont_data))
    return wordlist

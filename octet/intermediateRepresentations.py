from copy import copy
from collections import defaultdict
from octet.pulseBinarization import pulse, Spline, Discrete
from octet.encodingParameters import CLKFREQ


class PulseException(Exception):
    pass


class CollisionException(Exception):
    pass


class CircuitCompilerException(Exception):
    pass


def make_list_hashable(param):
    if isinstance(param, list):
        return Discrete(param)
    elif isinstance(param, tuple):
        return Spline(param)
    elif param is None:
        return 0
    else:
        return param

class ClockCycles(int):
    pass

def clock_cycles(dur):
    if isinstance(dur, ClockCycles):
        return dur
    return ClockCycles(dur*CLKFREQ)

def real_time(dur):
    if isinstance(dur, ClockCycles):
        return dur/CLKFREQ
    return dur

pd_cache = dict()

class PulseData:
    nonetypes = ['freq0', 'phase0', 'amp0', 'freq1', 'phase1', 'amp1', 'framerot0', 'framerot1']

    def __init__(self, channel, dur,
                 freq0=None, phase0=None, amp0=None,
                 freq1=None, phase1=None, amp1=None,
                 waittrig=False, sync_mask=0, enable_mask=0, fb_enable_mask=0,
                 framerot0=None, framerot1=None, apply_at_eof_mask=0, rst_frame_mask=0):
        self.channel = channel
        self.real_dur = dur
        self.dur = clock_cycles(dur)
        self.freq0 = make_list_hashable(freq0)
        self.phase0 = make_list_hashable(phase0)
        self.amp0 = make_list_hashable(amp0)
        self.freq1 = make_list_hashable(freq1)
        self.phase1 = make_list_hashable(phase1)
        self.amp1 = make_list_hashable(amp1)
        self.waittrig = waittrig
        self.sync_mask = sync_mask
        self.enable_mask = enable_mask
        self.fb_enable_mask = fb_enable_mask
        self.framerot0 = make_list_hashable(framerot0)
        self.framerot1 = make_list_hashable(framerot1)
        self.apply_at_eof_mask = apply_at_eof_mask
        self.rst_frame_mask = rst_frame_mask
        self.old_hash = None
        self.binary_data = None
        self.delay = 0

    def __repr__(self):
        return (f"ch: {self.channel} t: {self.dur} f0: {self.freq0}, p0: {self.phase0}, a0: {self.amp0}, "
                f"f1: {self.freq1}, p1: {self.phase1}, a1: {self.amp1}, fr: {self.framerot0}, fr2: {self.framerot1}, "
                f"wt:{self.waittrig}, sm: {self.sync_mask}, em: {self.enable_mask}, fe: {self.fb_enable_mask}"
                f"a: {self.apply_at_eof_mask}, r: {self.rst_frame_mask}, d: {self.delay}")

    def __eq__(self, other):
        if not isinstance(other, PulseData):
            return False
        return all(map(lambda attr: getattr(self, attr) == getattr(other, attr),
                       ['channel', 'dur', 'freq0', 'phase0', 'amp0', 'freq1', 'phase1',
                        'amp1', 'waittrig', 'sync_mask', 'enable_mask', 'fb_enable_mask',
                        'framerot0', 'framerot1', 'apply_at_eof_mask', 'rst_frame_mask', 'delay']))

    def __hash__(self):
        return hash((self.channel,
                     self.dur,
                     self.freq0,
                     self.phase0,
                     self.amp0,
                     self.freq1,
                     self.phase1,
                     self.amp1,
                     self.waittrig,
                     self.sync_mask,
                     self.enable_mask,
                     self.fb_enable_mask,
                     self.framerot0,
                     self.framerot1,
                     self.apply_at_eof_mask,
                     self.rst_frame_mask,
                     self.delay))

    def binarize(self, bypass=False, lru_cache=True):
        if lru_cache and hash(self) in pd_cache.keys():
            return pd_cache[hash(self)]
        DDS = self.channel
        dur = self.dur+self.delay
        freq0 = self.freq0
        phase0 = self.phase0
        amp0 = self.amp0
        freq1 = self.freq1
        phase1 = self.phase1
        amp1 = self.amp1
        waittrig = self.waittrig
        sync_mask = self.sync_mask
        enable_mask = self.enable_mask
        fb_enable_mask = self.fb_enable_mask
        framerot0 = self.framerot0
        framerot1 = self.framerot1
        apply_at_eof_mask = self.apply_at_eof_mask
        rst_frame_mask = self.rst_frame_mask
        self.binary_data = pulse(DDS, dur,
                                 freq0=freq0,
                                 phase0=phase0,
                                 amp0=amp0,
                                 waittrig=waittrig,
                                 freq1=freq1,
                                 phase1=phase1,
                                 amp1=amp1,
                                 sync_mask=sync_mask,
                                 enable_mask=enable_mask,
                                 fb_enable_mask=fb_enable_mask,
                                 framerot0=framerot0,
                                 framerot1=framerot1,
                                 apply_at_eof_mask=apply_at_eof_mask,
                                 rst_frame_mask=rst_frame_mask,
                                 bypass=bypass)
        pd_cache[hash(self)] = self.binary_data
        return self.binary_data

    @property
    def duration(self):
        return self.dur

    def bin2dur(self):
        if self.binary_data:
            print("B-----------------")
            bdatdict = defaultdict(int)
            for wrd in self.binary_data:
                bdat = int.from_bytes(wrd, byteorder='little', signed=False)
                nclks = (bdat>>160)&0xffffffffff
                dtype = (bdat>>253)
                bdatdict[dtype] += nclks#+3
                print(f"dtype {dtype}, clock cycles {nclks}")
            print("Total Times")
            for k,v in bdatdict.items():
                print(f" dtype {k}, total clk {v}")
            print("E-----------------")


def append_prepend_distribute(ch, pdl, dur, apd=0):
    if apd == 0: #append
        if len(pdl):
            if isinstance(pdl[-1], PulseData):
                new_pd = PulseData(ch, dur)
                for param in ['freq0', 'freq1']:
                    if type(getattr(pdl[-1], param)) in (Spline, Discrete):
                        setattr(new_pd, param, getattr(pdl[-1], param)[-1])
                pdl.append(new_pd)
            else:
                new_pd = PulseData(ch, dur)
                pdl[-1] = new_pd
        else:
            new_pd = PulseData(ch, dur)
            pdl.append(new_pd)
    elif apd == 1: #prepend
        if len(pdl):
            if isinstance(pdl[0], PulseData):
                new_pd = copy(pdl[0])
                new_pd.amp0 = 0
                new_pd.amp1 = 0
                new_pd.dur = dur
                #pdl.append(new_pd)
                pdl[0:0] = [new_pd]
            else:
                new_pd = PulseData(ch, dur)
                pdl[0:0] = [new_pd]
        else:
            new_pd = PulseData(ch, dur)
            pdl.append(new_pd)


class GateSlice:
    CHANNEL_NUM = 8

    def __init__(self, num_channels=None):
        self.channel_data = defaultdict(list)
        self.num_channels = num_channels or self.CHANNEL_NUM
        #self.repeats = 1

    def __repr__(self):
        retlist = []
        for k, v in self.channel_data.items():
            retlist.append(f"Channel {k}: {v}")
        return "\n".join(retlist)

    def merge(self, other):
        k1s = set(self.channel_data.keys())
        k2s = set(other.channel_data.keys())
        for k in k1s | k2s:
            if k in k1s:
                if k in k2s:
                    if self.channel_data[k] != other.channel_data[k]:
                        if self.channel_data[k] is None:
                            self.channel_data[k] = other.channel_data[k]
                        elif other.channel_data[k] is None:
                            continue
                        else:
                            raise CollisionException(f"Data does not match on channel {k}!")
            else:
                self.channel_data[k] = other.channel_data[k]

    def append(self, other):
        for k in other.channel_data.keys():
            self.channel_data[k].extend(other.channel_data[k])

    def __add__(self, other):
        new_gate_slice = GateSlice()
        new_gate_slice.merge(self)
        new_gate_slice.make_durations_equal()
        new_gate_slice.append(other)
        new_gate_slice.make_durations_equal()
        return new_gate_slice

    def __radd__(self, other):
        new_gate_slice = GateSlice()
        new_gate_slice.merge(other)
        new_gate_slice.make_durations_equal()
        new_gate_slice.append(self)
        new_gate_slice.make_durations_equal()
        return new_gate_slice

    def make_lengths_equal(self):
        channel_lengths = []
        for k in range(self.num_channels):
            channel_lengths.append(len(self.channel_data[k]))
        max_channel_length = max(channel_lengths)
        for k in range(self.num_channels):
            if channel_lengths[k] < max_channel_length:
                self.channel_data[k].extend([None]*(max_channel_length - channel_lengths[k]))

    def flatten_common_single(self, k):
        if len(self.channel_data[k])>1:
            for i, (pd1, pd2) in enumerate(zip(self.channel_data[k][:-1], self.channel_data[k][1:])):
                pd1_dag = copy(pd1)
                pd1_dag.dur = 1
                pd2_dag = copy(pd2)
                pd2_dag.dur = 1
                if pd1_dag == pd2_dag and not pd1_dag.waittrig:
                    pd1_dag.dur = ClockCycles(pd1.dur+pd2.dur)
                    self.channel_data[k][i:i+2] = [pd1_dag]
                    return True
        return False

    def flatten_common(self):
        for k in range(self.num_channels):
            while self.flatten_common_single(k):
                pass

    def make_durations_equal(self):
        if self.channel_data:
            durations = []
            for k in range(self.num_channels):
                current_duration = ClockCycles(0)
                for d in self.channel_data[k]:
                    if isinstance(d, PulseData):
                        current_duration += d.duration
                durations.append(current_duration)
            max_duration = max(durations)
            for k in range(self.num_channels):
                if durations[k] < max_duration:
                    append_prepend_distribute(k, self.channel_data[k], ClockCycles(max_duration-durations[k]), apd=0)
        return self

    def print_times(self, realtime=True):
        for k in range(self.num_channels):
            total_dur = ClockCycles(0)
            dur_list = []
            for pd in self.channel_data[k]:
                if isinstance(pd, PulseData):
                    total_dur += pd.duration
                    if realtime:
                        dur_list.append(pd.duration)
                    else:
                        dur_list.append(pd.dur)
                    pd.bin2dur()
            if realtime:
                print(f"Channel {k}; Total Duration: {real_time(ClockCycles(total_dur))}; Durations: {dur_list}")
            else:
                print(f"Channel {k}; Total Duration: {total_dur}; Durations: {dur_list}")


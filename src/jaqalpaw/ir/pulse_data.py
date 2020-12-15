from collections import defaultdict

from jaqalpaw.bytecode.pulse_binarization import pulse
from jaqalpaw.utilities.helper_functions import clock_cycles, make_list_hashable

pd_cache = dict()


class PulseData:
    nonetypes = [
        "freq0",
        "phase0",
        "amp0",
        "freq1",
        "phase1",
        "amp1",
        "framerot0",
        "framerot1",
    ]

    def __init__(
        self,
        channel,
        dur,
        freq0=None,
        phase0=None,
        amp0=None,
        freq1=None,
        phase1=None,
        amp1=None,
        waittrig=False,
        sync_mask=0,
        enable_mask=0,
        fb_enable_mask=0,
        framerot0=None,
        framerot1=None,
        apply_at_eof_mask=0,
        rst_frame_mask=0,
    ):
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
        return (
            f"ch: {self.channel} t: {self.dur} f0: {self.freq0}, p0: {self.phase0}, a0: {self.amp0}, "
            f"f1: {self.freq1}, p1: {self.phase1}, a1: {self.amp1}, fr: {self.framerot0}, fr2: {self.framerot1}, "
            f"wt:{self.waittrig}, sm: {self.sync_mask}, em: {self.enable_mask}, fe: {self.fb_enable_mask}"
            f"a: {self.apply_at_eof_mask}, r: {self.rst_frame_mask}, d: {self.delay}"
        )

    def __eq__(self, other):
        if not isinstance(other, PulseData):
            return False
        return all(
            map(
                lambda attr: getattr(self, attr) == getattr(other, attr),
                [
                    "channel",
                    "dur",
                    "freq0",
                    "phase0",
                    "amp0",
                    "freq1",
                    "phase1",
                    "amp1",
                    "waittrig",
                    "sync_mask",
                    "enable_mask",
                    "fb_enable_mask",
                    "framerot0",
                    "framerot1",
                    "apply_at_eof_mask",
                    "rst_frame_mask",
                    "delay",
                ],
            )
        )

    def __hash__(self):
        return hash(
            (
                self.channel,
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
                self.delay,
            )
        )

    def binarize(self, bypass=False, lru_cache=True):
        if lru_cache and hash(self) in pd_cache.keys():
            return pd_cache[hash(self)]
        DDS = self.channel
        dur = self.dur + self.delay
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
        self.binary_data = pulse(
            DDS,
            dur,
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
            bypass=bypass,
        )
        pd_cache[hash(self)] = self.binary_data
        return self.binary_data

    @property
    def duration(self):
        return self.dur

    def bin_to_dur(self):
        if self.binary_data:
            print("B-----------------")
            bdatdict = defaultdict(int)
            for wrd in self.binary_data:
                bdat = int.from_bytes(wrd, byteorder="little", signed=False)
                nclks = (bdat >> 160) & 0xFFFFFFFFFF
                dtype = bdat >> 253
                bdatdict[dtype] += nclks  # +3
                print(f"dtype {dtype}, clock cycles {nclks}")
            print("Total Times")
            for k, v in bdatdict.items():
                print(f" dtype {k}, total clk {v}")
            print("E-----------------")

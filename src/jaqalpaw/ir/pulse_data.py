from collections import defaultdict
from functools import lru_cache

from jaqalpaw.bytecode.pulse_binarization import pulse
from jaqalpaw.utilities.helper_functions import make_list_hashable
from jaqalpaw.utilities.datatypes import to_clock_cycles
from jaqalpaw.utilities.parameters import CLKFREQ


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
        apply_at_end_mask=0,
        rst_frame_mask=0,
        fwd_frame0_mask=0,
        fwd_frame1_mask=0,
        inv_frame0_mask=0,
        inv_frame1_mask=0,
    ):
        self.channel = channel
        self.real_dur = dur
        self.dur = to_clock_cycles(dur, CLKFREQ)
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
        self.apply_at_end_mask = apply_at_end_mask
        self.rst_frame_mask = rst_frame_mask
        self.fwd_frame0_mask = fwd_frame0_mask
        self.fwd_frame1_mask = fwd_frame1_mask
        self.inv_frame0_mask = inv_frame0_mask
        self.inv_frame1_mask = inv_frame1_mask
        self.old_hash = None
        self.binary_data = None
        self.delay = 0

    def __repr__(self):
        return (
            f"ch: {self.channel} t: {self.dur} f0: {self.freq0}, p0: {self.phase0}, a0: {self.amp0}, "
            f"f1: {self.freq1}, p1: {self.phase1}, a1: {self.amp1}, fr: {self.framerot0}, fr2: {self.framerot1}, "
            f"wt:{self.waittrig}, sm: {self.sync_mask}, em: {self.enable_mask}, fe: {self.fb_enable_mask}"
            f"a: {self.apply_at_end_mask}, r: {self.rst_frame_mask}, d: {self.delay} fwd0: {self.fwd_frame0_mask}, "
            f"fwd1: {self.fwd_frame1_mask}, inv0: {self.inv_frame0_mask} inv1: {self.inv_frame1_mask}"
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
                    "apply_at_end_mask",
                    "rst_frame_mask",
                    "fwd_frame0_mask",
                    "fwd_frame1_mask",
                    "inv_frame0_mask",
                    "inv_frame1_mask",
                    "delay",
                ],
            )
        )

    def almost_equal(self, other):
        """almost_equal is used for verifying that the data is equivalent with
        the exception of duration. The intended use case here is for checking
        equivalency of counter-propagating pulses with different durations. The
        global beam will be used in both pulses, but only has physical effect
        when the corresponding individual beam is also on. So a mismatch in
        global beam durations is acceptable as long as all other parameters are
        the same. This becomes more difficult to test if discrete/spline
        modulation is used. In principle the data could overlap if the spline
        or discrete modulations are defined just right, but at this point it is
        more likely that the user is explicitly trying to make this work and
        would probably construct their gates in a way to strictly enforce one
        particular global function on the tone in question. This test is not
        limited to a particular channel, in case the global is mapped to a
        different output, or other possible use cases arise."""
        if not isinstance(other, PulseData):
            return False
        all_eq_but_duration = all(
            map(
                lambda attr: getattr(self, attr) == getattr(other, attr),
                [
                    "channel",
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
                    "apply_at_end_mask",
                    "rst_frame_mask",
                    "fwd_frame0_mask",
                    "fwd_frame1_mask",
                    "inv_frame0_mask",
                    "inv_frame1_mask",
                    "delay",
                ],
            )
        )
        if all_eq_but_duration:
            all_params_static = all(
                map(
                    lambda attr: not isinstance(getattr(self, attr), (list, tuple)),
                    [
                        "freq0",
                        "phase0",
                        "amp0",
                        "freq1",
                        "phase1",
                        "amp1",
                        "framerot0",
                        "framerot1",
                    ],
                )
            )
            if all_params_static:
                return True
        return False

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
                self.apply_at_end_mask,
                self.rst_frame_mask,
                self.fwd_frame0_mask,
                self.fwd_frame1_mask,
                self.inv_frame0_mask,
                self.inv_frame1_mask,
                self.delay,
            )
        )

    @lru_cache(maxsize=4096)
    def binarize(self, bypass=False):
        self.binary_data = pulse(
            self.channel,
            self.dur + self.delay,
            freq0=self.freq0,
            phase0=self.phase0,
            amp0=self.amp0,
            waittrig=self.waittrig,
            freq1=self.freq1,
            phase1=self.phase1,
            amp1=self.amp1,
            sync_mask=self.sync_mask,
            enable_mask=self.enable_mask,
            fb_enable_mask=self.fb_enable_mask,
            framerot0=self.framerot0,
            framerot1=self.framerot1,
            apply_at_end_mask=self.apply_at_end_mask,
            rst_frame_mask=self.rst_frame_mask,
            fwd_frame0_mask=self.fwd_frame0_mask,
            fwd_frame1_mask=self.fwd_frame1_mask,
            inv_frame0_mask=self.inv_frame0_mask,
            inv_frame1_mask=self.inv_frame1_mask,
            bypass=bypass,
        )
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

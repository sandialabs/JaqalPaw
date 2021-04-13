from collections import defaultdict
from copy import copy

from .pulse_data import PulseData
from .padding import append_prepend_distribute
from jaqalpaw.utilities.datatypes import ClockCycles, to_real_time
from jaqalpaw.utilities.exceptions import CollisionException
from jaqalpaw.utilities.parameters import CLKFREQ


class GateSlice:
    CHANNEL_NUM = 8

    def __init__(self, num_channels=None):
        self.channel_data = defaultdict(list)
        self.num_channels = num_channels or self.CHANNEL_NUM
        # self.repeats = 1

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
                            # There might be a collision on a channel in which
                            # everything is the same except for the duration.
                            # This will primarily show up when the global beam
                            # is used for two operations in parallel, but the
                            # excess time on the global beam is irrelevant for
                            # the shorter of the two individual pulses as long
                            # as the data on the global beam is the same and all
                            # of the parameters are constant. In principle this
                            # could be extended to handle equivalent spline
                            # coefficients for which there is an iterable in one
                            # of the parameters that is matched over the shorter
                            # duration, but this is more difficult and will be
                            # less likely to occur so it is currently ignored.
                            # Also, the current case is limited to lists of
                            # length == 1 since the intended overlap of other
                            # data might break down for multiple PulseData objs.
                            if (
                                len(self.channel_data[k]) == len(other.channel_data[k])
                                and len(self.channel_data[k]) == 1
                                and self.channel_data[k][0].almost_equal(
                                    other.channel_data[k][0]
                                )
                            ):
                                if (
                                    self.channel_data[k][0].dur
                                    < other.channel_data[k][0].dur
                                ):
                                    self.channel_data[k] = other.channel_data[k]
                            else:
                                raise CollisionException(
                                    f"Data does not match on channel {k}!"
                                )
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
                self.channel_data[k].extend(
                    [None] * (max_channel_length - channel_lengths[k])
                )

    def flatten_common_single(self, k):
        if len(self.channel_data[k]) > 1:
            for i, (pd1, pd2) in enumerate(
                zip(self.channel_data[k][:-1], self.channel_data[k][1:])
            ):
                pd1_dag = copy(pd1)
                pd1_dag.dur = 1
                pd2_dag = copy(pd2)
                pd2_dag.dur = 1
                if pd1_dag == pd2_dag and not pd1_dag.waittrig:
                    pd1_dag.dur = ClockCycles(pd1.dur + pd2.dur)
                    self.channel_data[k][i : i + 2] = [pd1_dag]
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
                    append_prepend_distribute(
                        k,
                        self.channel_data[k],
                        ClockCycles(max_duration - durations[k]),
                        apd=0,
                    )
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
                    pd.bin_to_dur()
            if realtime:
                print(
                    f"Channel {k}; Total Duration: {to_real_time(ClockCycles(total_dur), CLKFREQ)}; Durations: {dur_list}"
                )
            else:
                print(
                    f"Channel {k}; Total Duration: {total_dur}; Durations: {dur_list}"
                )

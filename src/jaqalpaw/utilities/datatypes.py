class ClockCycles(int):
    pass


def to_clock_cycles(dur, clkfreq):
    if isinstance(dur, ClockCycles):
        return dur
    return ClockCycles(dur * clkfreq)


def to_real_time(dur, clkfreq):
    if isinstance(dur, ClockCycles):
        return dur / clkfreq
    return dur


class Spline(tuple):
    pass


class Discrete(tuple):
    pass


class Loop(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeats = repeats


class Branch(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeats = repeats

    @property
    def maxlen(self):
        return max(map(len, self))


class Case(list):
    def __init__(self, *args, state=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state


class Parallel(list):
    pass


class Sequential(list):
    pass

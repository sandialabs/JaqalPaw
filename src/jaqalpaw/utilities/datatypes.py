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


class ModulationBaseClass(tuple):
    """ModulationBaseClass is used to distinguish hashes for tuples
    containing parameters for different modulation types"""

    def __hash__(self):
        return hash((type(self), super().__hash__()))


class Spline(ModulationBaseClass):
    pass


class Discrete(ModulationBaseClass):
    pass


class Mixed(ModulationBaseClass):
    pass


class Loop(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._repeats = repeats

    @property
    def repeats(self):
        return int(self._repeats)


class Branch(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)

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

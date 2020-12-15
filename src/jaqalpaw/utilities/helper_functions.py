from .datatypes import ClockCycles, Discrete, Spline
from .parameters import CLKFREQ


def make_list_hashable(param):
    if isinstance(param, list):
        return Discrete(param)
    elif isinstance(param, tuple):
        return Spline(param)
    elif param is None:
        return 0
    else:
        return param

def clock_cycles(dur):
    if isinstance(dur, ClockCycles):
        return dur
    return ClockCycles(dur*CLKFREQ)

def real_time(dur):
    if isinstance(dur, ClockCycles):
        return dur/CLKFREQ
    return dur

def delist(x):
    """For use in pulse command, just ensures that a single value
       or a list with a single value is simply treated as that value"""
    if isinstance(x, list):
        return x[0]
    else:
        return x



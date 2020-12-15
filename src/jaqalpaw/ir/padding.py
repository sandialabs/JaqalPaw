from copy import copy

from .pulse_data import PulseData
from jaqalpaw.utilities.datatypes import Spline, Discrete


def append_prepend_distribute(ch, pdl, dur, apd=0):
    if apd == 0:  # append
        if len(pdl):
            if isinstance(pdl[-1], PulseData):
                new_pd = PulseData(ch, dur)
                for param in ["freq0", "freq1"]:
                    if type(getattr(pdl[-1], param)) in (Spline, Discrete):
                        setattr(new_pd, param, getattr(pdl[-1], param)[-1])
                pdl.append(new_pd)
            else:
                new_pd = PulseData(ch, dur)
                pdl[-1] = new_pd
        else:
            new_pd = PulseData(ch, dur)
            pdl.append(new_pd)
    elif apd == 1:  # prepend
        if len(pdl):
            if isinstance(pdl[0], PulseData):
                new_pd = copy(pdl[0])
                new_pd.amp0 = 0
                new_pd.amp1 = 0
                new_pd.dur = dur
                pdl[0:0] = [new_pd]
            else:
                new_pd = PulseData(ch, dur)
                pdl[0:0] = [new_pd]
        else:
            new_pd = PulseData(ch, dur)
            pdl.append(new_pd)

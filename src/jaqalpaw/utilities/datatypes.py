
class ClockCycles(int):
    pass


class Spline(tuple):
    pass


class Discrete(tuple):
    pass


class Loop(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeats = repeats



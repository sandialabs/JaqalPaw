import timeit
import cProfile


class Benchmark:

    # How many times to run the run method
    inner_number = 100000
    # How many times to call setUp, followed by many invocations of run
    outer_number = 10

    def __init__(self):
        self.times = None

    def setUp(self):
        """Override to set up environment."""
        pass

    def tearDown(self):
        """Override to tear down environment."""
        pass

    def run(self):
        """Override to provide the function under test."""
        pass

    def __call__(self):
        self.run()

    def start(self):
        self.times = []
        i = 0
        while i < self.outer_number:
            try:
                time = timeit.timeit(self, setup=self.setUp, number=self.inner_number)
                i += 1
                self.times.append(time)
            except Exception as exc:
                print(f"Ignored Exception: {exc}")
            finally:
                self.tearDown()

    def profile(self):
        self.setUp()
        try:
            cProfile.runctx("self.run()", globals(), locals(), sort="cumtime")
        finally:
            self.tearDown()

    @property
    def time(self):
        return sum(self.times) / len(self.times)

    @property
    def min_time(self):
        return min(self.times)

    @property
    def max_time(self):
        return max(self.times)

    def report(self):
        return (
            f"{type(self).__name__}: {self.time} ({self.min_time}, {self.max_time}) s"
        )


def run_benchmarks(benchmarks):
    print("<benchmark>: <avg-time> (<min-time>, <max-time>) s")
    for bm_cls in benchmarks:
        bm = bm_cls()
        bm.start()
        print(bm.report())


def profile_benchmarks(benchmarks):
    print("<benchmark>: <avg-time> (<min-time>, <max-time>) s")
    for bm_cls in benchmarks:
        bm = bm_cls()
        bm.profile()

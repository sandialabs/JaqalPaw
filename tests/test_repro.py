import os
from pathlib import Path

import unittest, pytest
from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler


examples_dir = Path("examples")


def write_test_wvf_file(filename):
    def inner(self):
        cc = CircuitCompiler(file=filename.with_suffix(".jaqal"))
        cc.compile()
        code = b"".join(qqq for q in cc.bytecode(0xFF) for qq in q for qqq in qq)
        self.assertEqual(code, open(filename, "rb").read())

    return inner


def make_tests(d):
    for fname in os.listdir(d):
        fname = d / fname
        testname = fname.with_suffix("").name
        if fname.is_dir():
            yield from ((f"{testname}_{a}", b) for (a, b) in make_tests(fname))
        elif fname.suffix == ".wvf":
            yield (testname, write_test_wvf_file(fname))


class TestReproduceWaveform(unittest.TestCase):
    for testname, obj in make_tests(examples_dir):
        exec(f"test_{testname}_wvf = obj")
        del testname, obj

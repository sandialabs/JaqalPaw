import unittest, os
from pathlib import Path


class ReproductionTester(unittest.TestCase):
    def test_jaqal(self):
        from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler

        examples_dir = Path("examples")

        for filename in os.listdir(examples_dir):
            filename = examples_dir / Path(filename)
            if not filename.suffix == ".wvf":
                continue

            cc = CircuitCompiler(file=filename.with_suffix(".jql"))
            cc.compile()
            code = b"".join(qqq for q in cc.bytecode(0xFF) for qq in q for qqq in qq)
            self.assertEqual(code, open(filename, "rb").read())

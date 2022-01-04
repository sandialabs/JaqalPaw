# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import sys, argparse, time


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        prog="jaqalpaw-emulate",
        description="Execute a Jaqal program and show the generated waveform",
    )
    parser.add_argument(
        "filename",
        default=None,
        nargs="?",
        help="Jaqal file to execute (default to reading from stdin).",
    )
    parser.add_argument(
        "--suppres-plot",
        "-s",
        dest="suppress",
        action="store_true",
        help="Do not plot waveform output.",
    )
    parser.add_argument(
        "--time",
        "-t",
        dest="time",
        action="store_true",
        help="Measure the time to generate the output waveform.",
    )
    parser.add_argument(
        "--debug-traces",
        "-d",
        dest="debug",
        action="store_true",
        help="Automatically invoke the post-mortem debugger on exception",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        action="store_true",
        help="Emit waveform output to stdout.",
    )

    ns = parser.parse_args(argv)

    if ns.filename:
        txt = None
        filename = ns.filename
    else:
        txt = sys.stdin.read()
        filename = None

    from .compiler.jaqal_compiler import CircuitCompiler

    try:
        start = time.time()
        cc = CircuitCompiler(file=filename, code_literal=txt)
        cc.compile()
        code = b"".join(qqq for q in cc.bytecode(0xFF) for qq in q for qqq in qq)
        stop = time.time()
    except Exception as ex:
        if ns.debug:
            import pdb, traceback

            traceback.print_exc()
            _, _, tb = sys.exc_info()
            pdb.post_mortem(tb)
            return 1
        else:
            print(f"Error during parsing: {type(ex).__name__}: {ex}", file=sys.stderr)
            return 1

    if not ns.suppress:
        from .emulator.firmware_emulator import plot_octet_emulator_output

        plot_octet_emulator_output(code)

    if ns.time:
        sys.stderr.write(f"time: {stop-start}\n")

    if ns.output:
        sys.stdout.buffer.write(code)

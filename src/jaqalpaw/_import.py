# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from jaqalpaq._import import jaqal_import


def get_jaqal_pulses(jaqal_module, jaqal_filename=None):
    jp = jaqal_import(str(jaqal_module), "jaqal_pulses", jaqal_filename=jaqal_filename)
    return jp.GatePulses()

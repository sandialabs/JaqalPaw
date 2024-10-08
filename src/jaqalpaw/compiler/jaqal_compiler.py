import numpy
import numpy as np
from pathlib import Path
from collections import Counter as multiset, defaultdict
import logging
from itertools import chain, zip_longest
from functools import reduce
from enum import IntEnum
import collections

from jaqalpaw.ir.circuit_constructor import CircuitConstructor
from jaqalpaw.ir.gate_slice import GateSlice
from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.bytecode.lut_programming import (
    program_PLUT,
    program_SLUT,
    program_GLUT,
    gate_sequence_bytes,
)
from jaqalpaw.compiler.time_ordering import timesort_bytelist
from jaqalpaw.utilities.datatypes import Loop, to_clock_cycles, Branch, Case
from jaqalpaw.utilities.exceptions import CircuitCompilerException, LUTOverflowException
from jaqalpaw.utilities.parameters import CLKFREQ
from jaqalpaw.bytecode.encoding_parameters import (
    ANCILLA_ADDR_START,
    ANCILLA_ADDR_STOP,
    ANCILLA_COMPILER_TAG_BIT,
    ANCILLA_STATE_LSB,
    CHANNELS_PER_BOARD,
    DMA_MUX_LSB,
    ENABLE_MLUT_PACKING,
    GLUTW,
    GPRGW,
    MODTYPE_LSB,
    PLUTW,
    PROG_MODE_LSB,
    SLUTW,
    VERSION,
    VIRTUAL_ANCILLA_TAG_BIT,
)
from ..ir.circuit_constructor_visitor import populate_gate_slice
from jaqalpaw.bytecode.binary_conversion import int_to_bytes, bytes_to_int

# We basically want a large number here. BRANCH_PRIORITY_OFFSET is an
# artificial correction to the number of occurrences of a gate, which is used
# to prioritize programming. By introducing a large offset, gates in branch
# sequences are given the highest priority, followed by gates that occur with
# the highest frequency. In the event of an overflow, this will allow gates to
# be interleaved with compressed sequences and direct streaming data in which
# case branch sequences MUST be compressed to work with the ancilla bitmask
BRANCH_PRIORITY_OFFSET = 1e12

flatten = lambda x: [y for l in x for y in l]
group_adjacent = lambda a, k: zip(*([iter(a)] * k))
tree = lambda: defaultdict(tree)

# fmt: off
# ######################################################## #
# ---------- Convert GateSlice IR to Bytecode ------------ #
# ######################################################## #

class GateletOptimizationType(IntEnum):
    NoOptimization = 0
    OptimizeIfNecessary = 1
    ExhaustiveOptimization = 2


class LUTOverflowHandling(IntEnum):
    """How to respond when LUT capacity is exceeded. If RecursiveCompilation is
    used, dynamically split code and do partial recompilation/reprogramming. If
    RaiseException is used, then handling will be passed to the user. In this
    case, the user may prefer to request streaming bytecode instead, however
    this is not assumed since the upload speed/approach may be impacted."""
    RaiseException = 0
    RecursiveCompilation = 1
    StreamDataDirectly = 2
    InterleavedStream = 3


class CircuitCompiler(CircuitConstructor):
    """Compiles the bytecode to be uploaded to the Octet from
    the intermediate representation layer of GateSlice objects"""

    def __init__(
        self,
        file=None,
        num_channels=8,
        override_dict=None,
        pd_override_dict=None,
        pulse_definition=None,
        global_delay=None,
        code_literal=None,
        slice_list=None,
        gatelet_optimization=GateletOptimizationType.OptimizeIfNecessary,
        lut_overflow_handling=LUTOverflowHandling.InterleavedStream,
    ):
        super().__init__(num_channels, pulse_definition)
        self.file = file
        self.code_literal = code_literal
        self.gatelet_optimization = gatelet_optimization
        self.lut_overflow_handling = lut_overflow_handling
        self.max_gatelet_opt_iterations = 10
        self.gatelet_opt_fitness_target = 0
        self.validate_gatelet_optimization = False  # enable for debugging
        self.binary_data = defaultdict(list)
        self.unique_gates = defaultdict(dict)
        self.gate_hash_recurrence = defaultdict(lambda: defaultdict(int))
        self.gate_sequence_hashes = defaultdict(list)
        self.gate_sequence_ids = defaultdict(list)
        self.ordered_gate_identifiers = dict()
        self.final_byte_dict = defaultdict(list)
        self.programming_data = list()
        self.sequence_data = list()
        self.PLUT_data = defaultdict(list)
        self.MMAP_data = defaultdict(dict)
        self.GLUT_data = defaultdict(dict)
        self.PLUT_bin = defaultdict(list)
        self.MMAP_bin = defaultdict(list)
        self.GLUT_bin = defaultdict(list)
        self.GSEQ_bin = defaultdict(list)
        self.override_dict = override_dict
        self.pd_override_dict = pd_override_dict
        self.compiled = False
        self.delay_settings = None
        self.set_global_delay(global_delay)
        self.initialize_gate_name = "prepare_all"
        self.slice_list = slice_list
        if slice_list is None:
            self.import_gate_pulses()
        self.cclist = []
        self.flattened_slice_list = None
        self.boardseqdata = None
        self.circuit_has_branches = False

    @property
    def num_boards(self):
        return ((self.channel_num-1)//CHANNELS_PER_BOARD)+1

    def set_global_delay(self, global_delay=None):
        if global_delay is None:
            self.delay_settings = None
        elif isinstance(global_delay, collections.abc.Iterable):
            shortest_time = min(0,min(global_delay))
            default_delay = abs(min(0,shortest_time))
            self.delay_settings = defaultdict(
                lambda: to_clock_cycles(default_delay, CLKFREQ)
            )
            for i,d in enumerate(global_delay):
                self.delay_settings[i] = to_clock_cycles(d - shortest_time, CLKFREQ)
        else:
            default_delay = 0
            if global_delay < 0:
                default_delay = -global_delay
                global_delay = 0
            self.delay_settings = defaultdict(
                lambda: to_clock_cycles(default_delay, CLKFREQ)
            )
            self.delay_settings[0] = to_clock_cycles(global_delay, CLKFREQ)

    def recursive_append_and_expand(self, slices, appendto):
        """Walk nested lists and expand Loops for bypass data"""
        for s in slices:
            if isinstance(s, Loop):
                for _ in range(s.repeats):
                    self.recursive_append(s, appendto)
            elif isinstance(s, list):
                self.recursive_append(s, appendto)
            else:
                appendto.append(s)

    def recursive_append(self, slices, appendto, flattened_list=None):
        """Walk nested lists but don't expand loops"""
        if flattened_list is None:
            flattened_list = []
            if self.flattened_slice_list is None:
                self.flattened_slice_list = flattened_list
        for s in slices:
            if isinstance(s, Loop):
                for _ in range(s.repeats):
                    self.recursive_append(s, appendto, flattened_list)
            elif isinstance(s, list):
                self.recursive_append(s, appendto, flattened_list)
            else:
                appendto.append(s)
                flattened_list.append(s)
        return flattened_list

    def binarize_circuit(self, bypass=False):
        """Generate binary representation of all PulseData objects.
        Used primarily"""
        if not self.compiled:
            self.construct_circuit(
                self.file,
                override_dict=self.override_dict,
                pd_override_dict=self.pd_override_dict,
            )
        self.binary_data = defaultdict(list)
        circ_main = GateSlice(num_channels=self.channel_num)
        self.recursive_append_and_expand(self.slice_list, circ_main)
        self.apply_delays(self.delay_settings, circ_main=circ_main)
        for ch, pd_list in circ_main.channel_data.items():
            for pd in pd_list:
                self.binary_data[ch].append(pd.binarize(bypass=bypass))
        return self.binary_data

    def apply_delays(self, delay_settings=None, circ_main=None):
        """Apply delay after all triggers to account for coarse delays between channels.
        Primarily, this function serves to match AOM turn on times in counter-propagating
        configurations, where the AOMs and associated electronics might not be matched.
        However, this function is designed to support independent delays for each channel."""
        if circ_main is None:
            circ_main = GateSlice(num_channels=self.channel_num)
            self.flattened_slice_list = self.recursive_append(self.slice_list, circ_main)
        if delay_settings is None:
            return
        for ch, pd_list in circ_main.channel_data.items():
            for pd in pd_list:
                if pd.waittrig:
                    pd.delay = delay_settings[ch]

    def streaming_data_from_luts(self, channels=None):
        if channels is None:
            channels = (1<<self.channel_num)-1
        sorted_bytelist = []
        bytelist = []
        for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
            for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num)):
                if (1<<ch) & channels:
                    for gid in self.gate_sequence_ids[ch]:
                        mstartaddr, mstopaddr = self.GLUT_data[ch][gid]
                        for maddr in range(mstartaddr, mstopaddr+1):
                            bytelist.append(int_to_bytes(
                                bytes_to_int(self.PLUT_data[ch][self.MMAP_data[ch][maddr]])
                              | (0b111<<PROG_MODE_LSB)
                            ))
            sorted_bytelist.append(timesort_bytelist(bytelist))
            bytelist.clear()
        return sorted_bytelist

    def subcircuit_streaming_data_from_luts(self, channels=None):
        if channels is None:
            channels = (1<<self.channel_num)-1
        sorted_subcircuit_bytelist = []
        sorted_bytelist = []
        bytelist = []
        gs_ids = self.generate_gate_sequence_subcircuits_inds_only()
        for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
            for scind in range(len(gs_ids[0])):
                for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num)):
                    if (1<<ch) & channels:
                        for gid in gs_ids[ch][scind]:
                            mstartaddr, mstopaddr = self.GLUT_data[ch][gid]
                            for maddr in range(mstartaddr, mstopaddr+1):
                                bytelist.append(int_to_bytes(
                                    bytes_to_int(self.PLUT_data[ch][self.MMAP_data[ch][maddr]])
                                  | (0b111<<PROG_MODE_LSB)
                                ))
                sorted_bytelist.append(timesort_bytelist(bytelist))
                bytelist.clear()
            sorted_subcircuit_bytelist.append(sorted_bytelist.copy())
            sorted_bytelist.clear()
        return sorted_subcircuit_bytelist

    def streaming_data(self, channels=None):
        """Generate the binary data for direct streaming (bypass mode)"""
        self.binarize_circuit(bypass=True)
        if channels is None:
            channels = (1<<self.channel_num)-1
        sorted_bytelist = []
        bytelist = []
        for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
            for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num)):
                if (1<<ch) & channels:
                    bytelist.extend(self.binary_data[ch])
            sorted_bytelist.append(timesort_bytelist(flatten(bytelist)))
            bytelist.clear()
        return sorted_bytelist

    def subcircuit_streaming_data(self, channel_mask=None):
        """Generate the binary data for direct streaming (bypass mode), broken into subcircuits"""
        streaming_bytelist_full = self.streaming_data(channel_mask)
        subcircuit_stream_data_full = []
        for bbind in range(self.num_boards):
            prepare_all_sbytes = self.streaming_prepare_all(
                    (channel_mask>>(bbind*CHANNELS_PER_BOARD))&((1<<CHANNELS_PER_BOARD)-1))
            streaming_bytelist = streaming_bytelist_full[bbind]
            subcircuit_stream_data = [[streaming_bytelist[0]]]
            streaming_bytelist = streaming_bytelist[1:]
            for _ in range(1000000):
                if (prepare_all_sbytes[0] in streaming_bytelist[1:]):
                    startind = streaming_bytelist[1:].index(prepare_all_sbytes[0])+1
                    stopind = startind + len(prepare_all_sbytes)
                    if streaming_bytelist[startind:stopind] == prepare_all_sbytes:
                        subcircuit_stream_data[-1].extend(streaming_bytelist[:startind])
                        streaming_bytelist = streaming_bytelist[startind:]
                        subcircuit_stream_data.append([])
                    else:
                        subcircuit_stream_data[-1].extend(streaming_bytelist[:startind])
                        streaming_bytelist = streaming_bytelist[startind+1:]
                else:
                    break
            else:
                raise CircuitCompilerException("error trying to split up subcircuits in streaming data, "
                                               "increase loop count to handle more than 1e6 subcircuits")
            subcircuit_stream_data[-1].extend(streaming_bytelist)
            subcircuit_stream_data_full.append(subcircuit_stream_data.copy())
        return subcircuit_stream_data_full

    def streaming_prepare_all(self, channels=None):
        """Generate the binary data for direct streaming (bypass mode) prepare_all gate"""
        prepare_all_prefix = "gate_"
        if not hasattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name):
            prepare_all_prefix = "macro_"
            if not hasattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name):
                raise CircuitCompilerException(
                f"Pulse definition has no gate named gate_{self.initialize_gate_name}"
                )
        if prepare_all_prefix == "macro_":
            gate_data = getattr(self.pulse_definition,
                                prepare_all_prefix + self.initialize_gate_name
                               )(self.channel_num)[0]
        else:
            gate_data = getattr(self.pulse_definition,
                                prepare_all_prefix + self.initialize_gate_name
                               )(self.channel_num)
        gslice = populate_gate_slice(gate_data, self.channel_num)
        self.apply_delays(self.delay_settings, circ_main=gslice)
        self.prepbin = defaultdict(list)
        for ch, pd_list in gslice.channel_data.items():
            for pd in pd_list:
                self.prepbin[ch].append(pd.binarize(bypass=True))
        if channels is None:
            channels = (1<<self.channel_num)-1
        bytelist = []
        for ch in range(channels.bit_length()):
            if (channels>>ch) & 1:
                bytelist.extend(self.prepbin[ch])
        sorted_bytelist = timesort_bytelist(flatten(bytelist))
        return sorted_bytelist

    def walk_slice(self, slice_obj, gate_hashes, reps=1, addr_offset=0):
        """Recursively walk through a list of slice objects, including Loops,
        Branches, Cases, down to the lowest GateSlice objects in order to map
        out the PulseData hashes that serve as gate sequence ids. The slice_obj
        input parameter is originally a list, which in the simplest case will
        contain GateSlice objects. Each GateSlice object has been pre-calculated
        to achieve equivalent time boundaries across all channels. The GateSlice
        encodes a unified, composite gate object for which a particular set of
        gates is run on all channels for the same time. Each GateSlice is
        represented as a dict of lists, where the key is the channel, and the
        list contains PulseData objects that comprise the gate to be run on said
        channel

                GateSlice_0    ,     GateSlice_1     ,     Gateslice_2
           |ch:0, [pd 0, ...]|   |ch:0, [pd 1, ...]|   |ch:0, [pd 2, ...]|
           |ch:1, [pd 0, ...]|   |ch:1, [pd 1, ...]|   |ch:1, [pd 2, ...]|
           |ch:2, [pd 0, ...]|   |ch:2, [pd 1, ...]|   |ch:2, [pd 2, ...]|

        The "gates" in this sense are abstract because they may contain data
        that simply pads out a time to make sure everything lines up later in
        the circuit, but it still needs an identifier in the LUTs. So the list
        of GateSlices is effectively transposed, where walk_slice traverses the
        list of pulse data objects and assigns them a unique gate name (based on
        a hash) independently for each channel:

         ch:0 -> [[pd 0,...],[pd 1,...],[pd 2,...]] --> [gate_0,gate_1,gate_2]
         ch:1 -> [[pd 0,...],[pd 1,...],[pd 2,...]] --> [gate_0,gate_0,gate_1]
         ch:2 -> [[pd 0,...],[pd 1,...],[pd 2,...]] --> [gate_1,gate_0,gate_0]

        Data redundancies may lead to duplicate gates. walk_slice assigns an
        arbitrary identifier to a list of PulseData objects ([pd 0,...]) based
        on the hash of a tuple of that list. If the hash is repeated, then the
        existing gate identifier is appended to the sequence of gate hashes that
        will be used for encoding the gate sequence to be read out of the LUTs.

        Because other structures are used for specific purposes in conjunction
        with GateSlice objects, they are treated in different ways depending on
        the object type. For example, a Loop object only needs to be traversed
        once to calculate gate hashes, and the sequence of hashes will then be
        repeated. Branches will contain different Case objects, which contain
        additional address offset information used for programming the GLUT.

        Since these objects can be nested, but all contain GateSlice objects at
        the lowest level, walk_slice does a recursive descent to collect all
        unique gate information, the associated gate hash, and the linear
        sequence of gates to be sequenced in terms of their hashes. The hash
        sequence will mutate the gate_hashes input object, which is treated
        differently for each object type but is always a defaultdict(list).

        The other input arguments control the state of the current walk_slice
        call during recursion. reps sets the full number of repetitions for
        each gate, which accounts for nested loops, in order to track frequency.
        addr_offset is used to change the address MSBs for a particular Case
        condition in a Branch statement. Branch statements break the linearity
        of a circuit, and must account for multiple sequences that depend on an
        external hardware input. walk_slice will return a Branch class to
        represent the various hash sequences and their associated address
        offsets for later use by extract_gates in order to appropriately map the
        different branch cases to different memory regions. However, the gate
        uniqueness, minimal gate information, and the order in which they're
        sequenced is captured by walk_slice.
        """
        if isinstance(slice_obj, GateSlice):
            for ch, gate_pd_list in slice_obj.channel_data.items():
                # We've arrived at a GateSlice (which is type defaultdict(list))
                # Each key is a channel, and each list contains the binarized
                # pulse data associated with the "gate" to be run on each
                # channel for this slice. Hashes of each gate are calculated
                # and used to give the gates a unique fingerprint, then the
                # gate data is stored in self.unique_gates, the number of calls
                # to the gate is tracked in self.gate_hash_recurrence and
                # the sequence of gates is tracked as a list of gate hashes
                # which are later sorted and given specific addresses in the
                # gate sequencer LUTs.
                new_key = hash(tuple(gate_pd_list))
                # Taking the hash of a tuple adds a factor of 2 speedup for the
                # whole compilation. However, this leads to a small (~1/3e11)
                # chance of a hash collision. The setdefault method in
                # conjunction with the assertion gives comparable performance
                # but with hash collision detection. If a hash collision occurs,
                # the hash() call in new_key can be removed.
                gate_pd_ref = self.unique_gates[ch].setdefault(new_key, gate_pd_list)
                assert gate_pd_ref == gate_pd_list
                self.gate_hash_recurrence[ch][new_key] += reps
                # gate_hashes stores a list of sequential hashes that need to be
                # run for a gate sequence. However, because walk_slice might be
                # recursing from within a Branch/Case, the address offset is
                # tracked separately so the mapping of keys for a branch can be
                # handled correctly
                gate_hashes[ch].append((addr_offset, new_key))
        elif isinstance(slice_obj, Loop):
            # Loops contain data that is highly redundant, and only needs to be
            # walked once to acquire unique gate information.
            inner_gate_hashes = defaultdict(list)
            for loop_block in slice_obj:
                self.walk_slice(
                    loop_block,
                    gate_hashes=inner_gate_hashes,
                    reps=slice_obj.repeats * reps,
                    addr_offset=addr_offset,
                )
            for k, v in inner_gate_hashes.items():
                # only repeat the gate ids after walking a Loop
                gate_hashes[k].extend(v * slice_obj.repeats)
        elif isinstance(slice_obj, Branch):
            # Branches require a different return type to indicate
            # that extra programming steps must be performed.
            inner_gate_hashes = defaultdict(list)
            for case in slice_obj:
                self.walk_slice(
                    case,
                    gate_hashes=inner_gate_hashes,
                    reps=reps+BRANCH_PRIORITY_OFFSET,
                    addr_offset=addr_offset,
                )
            for k, v in inner_gate_hashes.items():
                gate_hashes[k].append(Branch(v))
        elif isinstance(slice_obj, Case):
            # Cases are blocks within a branch statement that have an explicit
            # state tied to the particular sequence. For example, in the event
            # of an ancilla readout of '01', the gates in the block need to be
            # stored at a different offset address corresponding to the '01'
            # measurement. Thus the gates in a case block need to all have the
            # correct address offset, which is given by the particular state
            # of the readout, and must affect all gates within the block. The
            # address is modified by bit shifting the programming address by
            # ANCILLA_STATE_LSB, which might need to change if the number of
            # ancilla qubits changes.
            inner_gate_hashes = defaultdict(list)
            for case_block in slice_obj:
                self.walk_slice(
                    case_block,
                    gate_hashes=inner_gate_hashes,
                    reps=reps,
                    addr_offset=slice_obj.state << ANCILLA_STATE_LSB,
                )
            for k, v in inner_gate_hashes.items():
                gate_hashes[k].append(v)
        else:
            # This is just a list, and we just need another layer of recursion.
            for nested_slice_obj in slice_obj:
                self.walk_slice(
                    nested_slice_obj,
                    gate_hashes=gate_hashes,
                    reps=reps,
                    addr_offset=addr_offset,
                )

    def extract_gates(self):
        """Define gate identifiers (GLUT addresses) for LUT programming on each
        channel. The walk_slice call compresses gatesdown to unique data, tags
        them with a hash, and mutates the self.gate_sequence_hashes variable so
        it contains a sequence of hashes to be run. walk_slice also sets data
        in self.gate_hash_recurrence to check for frequency of calls to a
        particular gate."""
        branch_index_counter = defaultdict(int)
        self.branches = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.unique_gates = defaultdict(dict)
        self.gate_hash_recurrence = defaultdict(lambda: defaultdict(int))
        self.gate_sequence_hashes = defaultdict(list)
        self.gate_sequence_ids = defaultdict(list)
        self.walk_slice(self.slice_list, gate_hashes=self.gate_sequence_hashes)
        self.ordered_gate_identifiers = dict()
        for ch in range(self.channel_num):
            self.ordered_gate_identifiers[ch] = dict()
            # Generate contiguous addresses for gate ids (or GLUT addresses) for
            # sequencing gates. They are ordered by recurrence, so gates that
            # are called most often are given the lowest addresses. This is done
            # for each channel, and stored in self.ordered_gate_identifiers to
            # tie the numeric ID to a gate hash.
            for numeric_gate_id, (gate_hash, ncalls) in enumerate(
                sorted(
                    self.gate_hash_recurrence[ch].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ):
                self.ordered_gate_identifiers[ch][numeric_gate_id] = gate_hash
            # Create a temporary inverted dict that maps hash -> gate id
            inverted_ordered_gids = {
                v: k for k, v in self.ordered_gate_identifiers[ch].items()
            }
            # Create a numeric gate sequence, which will be the form used for
            # generating the gate sequence bytecode. Branches need special
            # consideration now, and a secondary mapping needs to be made for
            # different cases. For a branch, the streamed sequence must be
            # identical for all cases. The easiest solution is to stream in a
            # linear sequence of gate ids (0,1,2,...). These will naturally be
            # differentiated from standard streaming words because the upper
            # address bits will modify the gate in the LUT. This is a problem
            # when handling Cases in which the readout is all zero, because
            # there is nothing to distinguish between a standard gate sequence
            # and an ancilla-based sequence. Thus the address is tagged with
            # ANCILLA_COMPILER_TAG_BIT, which will typically correspond to the
            # MSB of the GLUT address used for programming. This bit can be set
            # to default to 1 for the hardware via the Octet interface and will
            # differentiate between an ancilla readout sequence and a standard
            # gate sequence. For multiple branch statements, there will likely
            # be collisions with the previous branches in the upper part of the
            # GLUT address. So the gate ids must maintain a linear sequence that
            # is a continuation of the previous sequences. In other words for
            # 3 branches, each with 3 gates, the gate ids streamed for those
            # branches will be
            #
            #                   (0,1,2), (3,4,5), (6,7,8)
            #
            # If a branch has redundancies that match all possible cases, then
            # a linear sequence won't be necessary. In other words, the first
            # sequence could have a form of (0,1,0), in which case the second
            # sequence above could start from a lower index -> (2,3,4) or a
            # sequence could be reused all together. However, this optimization
            # has not yet been worked in.
            self.gate_sequence_ids[ch] = []
            for hash_or_branch in self.gate_sequence_hashes[ch]:
                # The list of sequential hashes has a special exception where
                # the return type is a branch, because a branch contains a
                # collection of hashes that need to be equivalent across all
                # possible ancilla states.
                if isinstance(hash_or_branch, Branch):
                    self.circuit_has_branches = True
                    branch_gate_sequence_ids = []
                    for case_index, case_gate_hashes in enumerate(hash_or_branch):
                        for sub_gate_id, (offset_addr, gate_hash) in enumerate(
                            case_gate_hashes
                        ):
                            if sub_gate_id == 0:
                                initlen = sum(
                                    len(self.branches[ch][branch_idx][offset_addr])
                                    for branch_idx in range(branch_index_counter[ch])
                                )
                            if case_index == 0:
                                branch_gate_sequence_ids.append(
                                    (sub_gate_id + initlen)
                                    | (1 << VIRTUAL_ANCILLA_TAG_BIT)
                                )
                            self.branches[ch][branch_index_counter[ch]][
                                offset_addr
                            ].append(inverted_ordered_gids[gate_hash])
                    self.gate_sequence_ids[ch].extend(branch_gate_sequence_ids)
                    branch_index_counter[ch] += 1
                else:
                    self.gate_sequence_ids[ch].append(
                        inverted_ordered_gids[hash_or_branch[1]]
                    )

    def generate_lookup_tables(self):
        """Construct the LUT data in an intermediate representation. The outputs
        are in a human readable format with the exception of the raw pulse data"""
        self.PLUT_data = defaultdict(list)
        self.MMAP_data = defaultdict(dict)
        self.GLUT_data = defaultdict(dict)
        plut_index_scratch = {}
        for ch in range(self.channel_num):
            gid = 0
            addr = 0
            plut_index_scratch.clear()
            for k, v in sorted(self.ordered_gate_identifiers[ch].items()):
                gate_start_addr = addr
                for pd in self.unique_gates[ch][v]:
                    pd_bin = pd.binarize()
                    for pdb in pd_bin:
                        if pdb not in plut_index_scratch:
                            plut_index_scratch[pdb] = len(self.PLUT_data[ch])
                            self.PLUT_data[ch].append(pdb)
                        self.MMAP_data[ch][addr] = plut_index_scratch[pdb]
                        addr += 1
                gate_end_addr = addr - 1
                self.GLUT_data[ch][gid] = (gate_start_addr, gate_end_addr)
                gid += 1
        for ch in range(self.channel_num):
            startind = 0
            for branch in self.branches[ch].values():
                maxlen = 0
                for state, glist in branch.items():
                    maxlen = max(len(glist), maxlen)
                    gind = 0
                    for subgid in glist:
                        self.GLUT_data[ch][
                            (startind + state + gind) | (1 << VIRTUAL_ANCILLA_TAG_BIT)
                        ] = self.GLUT_data[ch][subgid]
                        gind += 1
                startind += maxlen
        if ENABLE_MLUT_PACKING:
            self.GLUT_data_unpacked = self.GLUT_data.copy()
            self.PLUT_data_unpacked = self.PLUT_data.copy()
            self.MMAP_data_unpacked = self.MMAP_data.copy()
            self.gate_sequence_ids_unpacked = self.gate_sequence_ids.copy()
            cglut = defaultdict(dict)
            cmlut = defaultdict(dict)
            cplut = defaultdict(list)
            gluttot,mluttot,pluttot = 0,0,0
            ngluttot,nmluttot,npluttot = 0,0,0
            cgseq = defaultdict(list)
            self.gateTransList = []
            self.gate_limit_list = []
            luts_exceeded = False
            for n in range(self.channel_num):
                _cglut, _cmlut, _cplut, _cgseq, gatetrans, gatelimit, bls, als = self.recast_lookup_table(
                    self.GLUT_data[n],
                    self.MMAP_data[n],
                    self.PLUT_data[n],
                    self.gate_sequence_ids[n],
                    chidx=n)
                cglut[n].update(_cglut)
                cmlut[n].update(_cmlut)
                cplut[n].extend(_cplut)
                cgseq[n].extend(_cgseq)
                self.gateTransList.append(gatetrans)
                self.gate_limit_list.append(gatelimit)
                gluttot += bls[0]
                mluttot += bls[1]
                pluttot += bls[2]
                ngluttot += als[0]
                nmluttot += als[1]
                npluttot += als[2]
                if gatelimit:
                    luts_exceeded = True
            logging.getLogger(__name__).debug(
                    f"Total GLUT: before {gluttot}, after {ngluttot}, ratio: {ngluttot/gluttot}")
            logging.getLogger(__name__).debug(
                    f"Total MLUT: before {mluttot}, after {nmluttot}, ratio: {nmluttot/mluttot}")
            logging.getLogger(__name__).debug(
                    f"Total PLUT: before {pluttot}, after {npluttot}, ratio: {npluttot/pluttot}")
            logging.getLogger(__name__).debug(
                    f"Total Savings: {(ngluttot+nmluttot+npluttot)/(gluttot+mluttot+pluttot)}")
            self.PLUT_data = cplut
            self.MMAP_data = cmlut
            self.GLUT_data = cglut
            self.gate_sequence_ids = cgseq
            return luts_exceeded
        else:
            for ch in range(self.channel_num):
                logging.getLogger(__name__).debug(f"LUT sizes channel: {ch}")
                logging.getLogger(__name__).debug(f"Total GLUT: {len(self.GLUT_data[ch])}")
                logging.getLogger(__name__).debug(f"Total MLUT: {len(self.MMAP_data[ch])}")
                logging.getLogger(__name__).debug(f"Total PLUT: {len(self.PLUT_data[ch])}")
        return False

    @staticmethod
    def PLUT256to216(data, include_mod_type=True):
        if VERSION == 2:
            return int_to_bytes(bytes_to_int(data) & ((1<<MODTYPE_LSB)-1))
        idata = bytes_to_int(data)
        ndata = idata & ((1<<200)-1)
        if include_mod_type:
            ndata |= (idata>>MODTYPE_LSB & 0b111) << 213
        ndata |= (idata>>248 & 0b11111) << 208
        ndata |= (idata>>224 & 0b11111) << 203
        ndata |= (idata>>218 & 0b11) << 201
        return int_to_bytes(ndata)

    def recast_lookup_table(self, glut, mlut, plut, gseq, chidx=0, exhaustive=0, maxiter=100, optimbound=0, smartoptim=True):
        """Remap data in MLUT/PLUT so that the MLUT stores N-hot routing"""
        nplutmap = dict()
        nroutmap = dict()
        nplutset = dict()
        ngsetmlut = dict()
        nmlut = dict()
        # get address and data of all elements of PLUT, filter out the routing
        # data and add to a set for tracking unique data
        for pido, pdato in enumerate(plut):
            nplutset[self.PLUT256to216(pdato,include_mod_type=False)] = \
                    int_to_bytes(bytes_to_int(pdato)&((1<<MODTYPE_LSB)-1))

        # convert the set to a list so we can reliably retrieve the data's associated index
        nplutls = tuple(nplutset.keys())
        nplutinds = {v:k for k,v in enumerate(nplutls)}

        if len(nplutls) > (1<<PLUTW) and self.lut_overflow_handling != LUTOverflowHandling.InterleavedStream:
            raise LUTOverflowException(f"PLUT filling exceeded in recast {self.lut_overflow_handling}")

        # create a dict that reindexes the original PLUT data based on the new
        # set. For each original plut address, also store the target
        # parameter/tone in a separate routing dictionary
        for pido, pdato in enumerate(plut):
            nplutmap[pido] = nplutinds[self.PLUT256to216(pdato,include_mod_type=False)]
            if VERSION==2:
                nroutmap[pido] = ((bytes_to_int(pdato)>>MODTYPE_LSB)&0xff).bit_length()-1
            else:
                nroutmap[pido] = (bytes_to_int(pdato)>>MODTYPE_LSB)&0b111

        # reconstruct GLUT and MLUT data based on the new PLUT representation.
        # Basically just walk the GLUT and convert the original address range
        # and associated MLUT values to a new representation. Because this new
        # representation uses N-hot encoding, the number of entries will be
        # reduced. So the new mlut address is generated using a counter
        # (nmaddrcnt), which does not increment as new routing data is added to
        # the same MLUT entry. Because a NOP used to be encoded in the PLUT as
        #     {0: (amp0, data), 1: (amp1, data), 2: (freq0, data), ...}
        # where data is all the same and only the routing information changes,
        # the new formalism will result in a single PLUT entry since routing
        # information was stripped out, and effectively the MLUT absorbs the
        # data. MLUT filling is still a problem though, so by giving up some
        # data bits in the packed representation at the fw level we can tag the
        # MLUT data with a one-hot routing mask. Before the MLUT and GLUT would
        # encode the above NOP as
        #     MLUT: {0: 0, 1: 1, 2: 2, ...}         GLUT: {0: (0,7)}
        # Now, the encoding will be
        #     MLUT: {0: (0b11111111, 0)}            GLUT: {0: (0,0)}
        # and an Rz gate would be
        #     MLUT: {0: (0b10111111, 0), 1: (0b01000000, 1) }  GLUT: {0: (0,1)}
        # Thus the MLUT data size increases from 12 to 20 bits and the encoding
        # is N-hot, the number of entries will differ from before.

        nmaddrcnt = 0
        glutrangereductionmap = dict()
        glutrangeremap = dict()
        # setup reverse glut mapping that accounts for branching
        rglut = defaultdict(list)

        PLUTWEXT = PLUTW+10
        gatelimit = (1<<GLUTW)
        badgates = set()
        for gn,gr in glut.items():
            rglut[gr].append(gn)
        # walk the original GLUT
        for gido, gdato in glut.items():
            if self.circuit_has_branches:
                if self.gatelet_optimization:
                    self.gatelet_optimization = 0
                    logging.getLogger(__name__).warning("gatelet optimization not yet allowed for circuits with branching")
            if gido > (1<<GLUTW):
                self.gatelet_optimization = 0
            if len(rglut[gdato]) > 1 and gdato in glutrangeremap:
                continue
            lmdatotrack = defaultdict(list)
            nmstartaddr = nmaddrcnt  # new GLUT address bounds need to be tracked independently
            localcount = 0
            # get the MLUT addresses
            for maddr in range(gdato[0],gdato[1]+1):
                mdato = mlut[maddr]  # old MLUT data (old PLUT addr)
                # check if remapped PLUT data has been encountered by looking at the remapped PLUT addr
                # if nplutmap[mdato] in lmdatoset:
                if nplutmap[mdato] in lmdatotrack:
                    if (nmlut[lmdatotrack[nplutmap[mdato]][-1]+nmstartaddr] & (1<<(PLUTWEXT+nroutmap[mdato]))):
                        # This entry has already been referenced, this is a repeated occurrence so we need to
                        # advance the address. Due to how sets are being used, we augment the data, and this
                        # gets filtered out during programming. The reps command is tracked separately from
                        # the address for this purpose.
                        lmdatotrack[nplutmap[mdato]].append(localcount)
                        nmlut[nmaddrcnt] = nplutmap[mdato] | 1<<(PLUTWEXT+nroutmap[mdato]) | (len(lmdatotrack[nplutmap[mdato]])-1)<<(PLUTWEXT+8)
                        nmaddrcnt += 1
                        localcount += 1
                    else:
                        # PLUT addr already used by current gate, just tag with additional routing data
                        nmlut[lmdatotrack[nplutmap[mdato]][-1]+nmstartaddr] |= (1<<(PLUTWEXT+nroutmap[mdato]))
                else:
                    # we're getting a new PLUT addr, store in new MLUT, tag with
                    # the initial routing data and advance the new MLUT address
                    lmdatotrack[nplutmap[mdato]].append(localcount)
                    nmlut[nmaddrcnt] = nplutmap[mdato] | 1<<(PLUTWEXT+nroutmap[mdato])
                    nmaddrcnt += 1
                    localcount += 1
            # the current gate has been translated in the new MLUT, but we aren't done yet
            # create a mapping of the old gate id (which has otherwise been perfectly valid)
            # and how it connects to the translated MLUT data.
            ngsetmlut[gido] = tuple(nmlut[nn] for nn in range(nmstartaddr,nmaddrcnt))
            # we can also store info about how the gate bounds were remapped,
            # but this will lose its utility soon and is only for inspection atm
            glutrangereductionmap[gido] = ((gdato, (nmstartaddr, nmaddrcnt-1)))
            glutrangeremap[gdato] = (nmstartaddr, nmaddrcnt-1)

        # Now we've remapped all of the gates into the new format. However,
        # there are likely redundancies for gates with one or two parameters
        # that change. For gates with a lot of identical data (such as Rz,
        # where most of the data is zero except the frame rotation), this means
        # we'll often be calling two words with one of them always the same. In
        # this case, we still see a reduction in both PLUT and MLUT data, but
        # if data differs on all parameters, then gates where only one or two
        # parameters change will still be filling up the MLUT (as before).
        # Since the new encoding cuts down on effective MLUT address width, the
        # repeated calls are worth handling.
        #
        # Here, we look for overlap in the set of PLUT addresses used by each
        # gate, if there is a lot of overlap with other gates, then we can
        # break the common data into a gatelet, and separately stream partial
        # gates. This will also be quite useful in reducing the amount of data
        # that needs to be transferred for direct streaming in the event that
        # we have a certain parameter that is changing regularly. Instead of
        # trying to dynamically reprogram, we can eliminate the additional
        # storage from a smaller set of parameters by passing gate data in as a
        # sideband, or just back-to-back with the single word. Streaming will
        # also be more viable in this scheme since NOP data can be broadcasted
        # to any/all channels with a single word.

        # Get the set of MLUT data for each gate
        S = tuple(ngsetmlut.values())
        gateTransList = None
        # optionally optimize MLUT storage by breaking gates into gatelets
        # in this case, redundant calls to a portion of a gate (which might
        # occur when gates containing a lot of unique data are called with
        # one parameter change, such as phase) can be optimized in the MLUT
        # by sequencing two gatelet identifiers. This will prevent the MLUT
        # from filling up if single qubit gates
        new_gseq_list = []
        if ((self.gatelet_optimization > GateletOptimizationType.OptimizeIfNecessary
            or (self.gatelet_optimization == GateletOptimizationType.OptimizeIfNecessary
                and sum(map(len,S))>(1<<SLUTW)))
            and len(glut) < (1<<GLUTW)):
            if self.validate_gatelet_optimization:
                Sinit = S
            if self.lut_overflow_handling == LUTOverflowHandling.InterleavedStream:
                maxiter = self.max_gatelet_opt_iterations
            else:
                maxiter = None
            nglutdist, gateTransList, mlutsize = distill_gatelets(ngsetmlut, 1<<SLUTW, maxiter=maxiter)
            if mlutsize > 1<<SLUTW and self.lut_overflow_handling != LUTOverflowHandling.InterleavedStream:
                raise LUTOverflowException("MLUT filling exceeded")
            iteritem = nglutdist.items()
            if self.validate_gatelet_optimization:
                verify_distillation(ngsetmlut,nglutdist,gateTransList)
            for oldgid in gseq:
                if oldgid not in badgates:
                    new_gseq_list.extend(gateTransList[oldgid])
        else:
            new_gseq_list = gseq
            iteritem = enumerate(S)

        glutfin = dict()
        mlutfin = dict()
        plutfin = tuple(nplutset.values())
        maddrctr = 0
        glut_filling_ok = True
        for gid, gdat in iteritem:
            if gid & (1<<VIRTUAL_ANCILLA_TAG_BIT):
                continue # these are handled in the next loop
            if gid >= 1<<GLUTW:
                if self.lut_overflow_handling != LUTOverflowHandling.InterleavedStream:
                    raise LUTOverflowException("GLUT filling exceeded")
                else:
                    if glut_filling_ok:
                        # just want to trigger the debug message once
                        logging.getLogger(__name__).debug(
                                f"GLUT filling exceeded at gate: {gid}, remaining gates will be interleaved")
                        glut_filling_ok = False
                    if gateTransList:
                        bad_trans_gate = False
                        for ogid, glist in gateTransList.items():
                            for ngid in glist:
                                if ngid >= 1<<GLUTW:
                                    bad_trans_gate = True
                                    badgates.add(ogid)
                        if bad_trans_gate:
                            continue
                    else:
                        badgates.add(gid)
                        continue
            maddrstart = maddrctr
            for maddr in gdat:
                if maddrctr >= 1<<SLUTW or (maddr&((1<<PLUTWEXT)-1)) >= 1<<PLUTW:
                    if maddrctr >= 1<<SLUTW:
                        logging.getLogger(__name__).debug(f"MLUT filling exceeded in gate: {gid}")
                    else:
                        logging.getLogger(__name__).debug(f"PLUT filling exceeded in gate: {gid}")
                    if gateTransList:
                        for ogid, glist in gateTransList.items():
                            if gid in glist:
                                badgates.add(ogid)
                    else:
                        badgates.add(gid)
                    break
                mlutfin[maddrctr] = (maddr&((1<<PLUTW)-1))|((maddr>>PLUTWEXT)<<PLUTW)
                maddrctr += 1
            else:
                glutfin[gid] = (maddrstart, maddrctr-1)
        for gid in glut.keys():
            if gid & (1<<VIRTUAL_ANCILLA_TAG_BIT):
                glutfin[gid] = glutrangeremap[glut[gid]]

        logging.getLogger(__name__).debug(f"FINAL LENGTHS: ch{chidx}")
        logging.getLogger(__name__).debug(
                f" GLUT: was {len(glut)} now {len(glutfin)} reduction {len(glutfin)/len(glut)}")
        logging.getLogger(__name__).debug(
                f" MLUT: was {len(mlut)} now {len(mlutfin)} reduction {len(mlutfin)/len(mlut)}")
        logging.getLogger(__name__).debug(
                f" PLUT: was {len(plut)} now {len(plutfin)} reduction {len(plutfin)/len(plut)}")

        return (
                glutfin,
                mlutfin,
                plutfin,
                new_gseq_list,
                gateTransList,
                badgates,
                (len(glut), len(mlut), len(plut)),
                (len(glutfin), len(mlutfin), len(plutfin)),
               )

    def generate_interleaved_data(self, channel_mask=None):
        """Combine compressed and directly-streamed data such that LUT
           overflows are handled by falling back on streaming as needed."""
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        self.PLUT_bin = defaultdict(list)
        self.MMAP_bin = defaultdict(list)
        self.GLUT_bin = defaultdict(list)
        self.GSEQ_bin = defaultdict(list)
        self.get_prepare_all_indices()
        partial_gs_ids = dict()
        for ch, gidlist in self.gate_sequence_ids_unpacked.items():
            partial_gs_ids[ch] = get_subcircuits(
                self.prepare_all_gids[ch], gidlist
            )
        num_subcircuits = len(partial_gs_ids[0])
        progvalid_list = []
        for scind in range(num_subcircuits):
            progvalid_list.append([True]*len(partial_gs_ids[0][scind]))
        for ch in range(self.channel_num):
            for scind in range(num_subcircuits):
                for idx, oldgid in enumerate(partial_gs_ids[ch][scind]):
                    if oldgid in self.gate_limit_list[ch]:
                        progvalid_list[scind][idx] = False
        for ch in range(self.channel_num):
            self.PLUT_bin[ch], errp = program_PLUT(
                {v: i for i, v in enumerate(self.PLUT_data[ch])}, ch
            )
            self.GLUT_bin[ch], errg = program_GLUT(self.GLUT_data[ch], ch)
            self.MMAP_bin[ch], errm = program_SLUT(self.MMAP_data[ch], ch)
        # Determine if programming data is valid across a slice by checking
        # validity across channels. If any channel has a gate that exceeds one
        # of the LUTs, then mark the slice as invalid.
        statelist = [[] for scind in range(num_subcircuits)]
        for scind in range(num_subcircuits):
            state = progvalid_list[scind][0]
            statectr = 0
            for pstate in progvalid_list[scind]:
                if pstate != state:
                    statelist[scind].append((state, statectr))
                    state = pstate
                    statectr = 0
                statectr += 1
            statelist[scind].append((state, statectr))
        gids_unpacked = partial_gs_ids.copy()

        # For each subcircuit, accumulate slice data based on the validity of
        # its programming data. If it's valid, append sequence data, otherwise
        # append direct streaming data. Even though an overflow might only
        # occur on a single channel for any given slice, we still stream the
        # entire slice. The reason for this is twofold:
        #   1) Asymmetry in the sequence data caused by gatelet distillation is
        #      likely to pose more issues when interleaving with direct
        #      streaming data. By defining strict time boundaries at each
        #      transition, we only have to worry about data asymmetry between
        #      boundaries.
        #   2) Streaming data is actually more efficient than sequenced data
        #      for NOPs with the broadcasted routing scheme. Since most of the
        #      sequenceable data that coincides with an overflow on another
        #      channel will likely be NOPs, we are guaranteed to only need a
        #      single word for NOPs on all channels that need one, as opposed
        #      to N words for sequencing NOPs. The sequenced data reduces to a
        #      single word if the gate identifiers are identical for the N
        #      channels, but this is highly subjective.
        board_subcircuit_sequence_data = [[] for _ in range(self.num_boards)]
        for scind in range(num_subcircuits):
            board_sequence_data = [[] for _ in range(self.num_boards)]
            cntsum = 0  # cntsum tracks a sliding window for gate ids
            for pstate, cnt in statelist[scind]:
                if pstate:  # programming data is valid, pack up sequence data
                    self.GSEQ_bin = defaultdict(list)
                    for ch in range(self.channel_num):
                        seqlist = []
                        # Iterate over the _unpacked_ gate ids (before gatelet
                        # distillation), then translate to distilled gates.
                        # This is because slice index validity was determined
                        # on the undistilled gates to ensure uniformity across
                        # channels, but gatelet asymmetry can now be worked
                        # back in.
                        for ogid in gids_unpacked[ch][scind][cntsum:cntsum+cnt]:
                            if self.gateTransList[ch] is not None:
                                for ngid in self.gateTransList[ch][ogid]:
                                    seqlist.append(ngid)
                            else:
                                seqlist.append(ogid)
                        self.GSEQ_bin[ch].extend(gate_sequence_bytes(seqlist, ch))
                    for bidx,bbind in enumerate(range(0, self.channel_num, CHANNELS_PER_BOARD)):
                        board_seq_dat = []
                        # raster through channels to interleave sequence data
                        for bindata in zip_longest(
                            *list(
                                self.GSEQ_bin[ch]
                                for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                                if (1 << ch) & channel_mask
                            ),
                            fillvalue=b"",
                        ):
                            if bindata:
                                board_seq_dat.extend(bindata)
                        # sequence data might match an a lot of channels, filter
                        # out routing data and compare content, then broadcast
                        newsbytes = []
                        pvd = defaultdict(int)
                        for chdat in group_adjacent(board_seq_dat,CHANNELS_PER_BOARD):
                            pvd = defaultdict(int)
                            for pv in map(bytes_to_int, chdat):
                                pvm = pv & ((1<<DMA_MUX_LSB)-1)
                                pvr = pv >> DMA_MUX_LSB
                                pvd[pvm] |= pvr
                            for pvk,pvv in pvd.items():
                                newsbytes.append(int_to_bytes(pvk|(pvv<<DMA_MUX_LSB)))
                        board_sequence_data[bidx].extend(newsbytes)
                else:  # programming data is invalid, feed in streaming data
                    sorted_bytelist = []
                    bytelist = []
                    for bidx, bbind in enumerate(range(0, self.channel_num, CHANNELS_PER_BOARD)):
                        bytelist = []
                        for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num)):
                            if (1<<ch) & channel_mask:
                                # In this case there is no need to worry about gatelet
                                # distillation. Rather, timesorting of streaming data now
                                # accounts for packing of shared parameter and channel data
                                # simultaneously, so we are just using the fully unpacked
                                # representation for now.
                                for gid in gids_unpacked[ch][scind][cntsum:cntsum+cnt]:
                                    mstartaddr, mstopaddr = self.GLUT_data_unpacked[ch][gid]
                                    for maddr in range(mstartaddr, mstopaddr+1):
                                        bytelist.append(int_to_bytes(
                                            bytes_to_int(self.PLUT_data_unpacked[ch][
                                                            self.MMAP_data_unpacked[ch][maddr]
                                                         ])
                                          | (0b111<<PROG_MODE_LSB)
                                        ))
                        board_sequence_data[bidx].extend(timesort_bytelist(bytelist))
                        bytelist.clear()
                cntsum += cnt
            for bn in range(self.num_boards):
                board_subcircuit_sequence_data[bn].append(board_sequence_data[bn])
        return board_subcircuit_sequence_data

    def generate_programming_data(self):
        """Convert the LUT programming IR representations to bytecode"""
        self.PLUT_bin = defaultdict(list)
        self.MMAP_bin = defaultdict(list)
        self.GLUT_bin = defaultdict(list)
        self.GSEQ_bin = defaultdict(list)
        badinds = []
        for ch in range(self.channel_num):
            self.PLUT_bin[ch], errp = program_PLUT(
                {v: i for i, v in enumerate(self.PLUT_data[ch])}, ch
            )
            self.GLUT_bin[ch], errg = program_GLUT(self.GLUT_data[ch], ch)
            self.MMAP_bin[ch], errm = program_SLUT(self.MMAP_data[ch], ch)
            self.GSEQ_bin[ch] = gate_sequence_bytes(self.gate_sequence_ids[ch], ch)
            if errp:
                if self.lut_overflow_handling == LUTOverflowHandling.RaiseException:
                    raise LUTOverflowException("PLUT filling exceeded")
                for i, d in enumerate(self.MMAP_data[ch].values()):
                    if d == errp:
                        indp = i
                        break
                else:
                    indp = 1 << 12
                for i, (gi, (gu, gl)) in enumerate(self.GLUT_data[ch].items()):
                    if gu >= indp or gl >= indp:
                        bgid = gi
                        break
                for i, gid in enumerate(self.gate_sequence_ids[ch]):
                    if gid == bgid:
                        inds = i
                        badinds.append(inds)
                        break
            if errm:
                if self.lut_overflow_handling == LUTOverflowHandling.RaiseException:
                    raise LUTOverflowException("MLUT filling exceeded")
                for i, a in enumerate(self.MMAP_data[ch]):
                    if a == errm:
                        indp = i
                        break
                else:
                    indp = 1 << 12
                for i, (gi, (gu, gl)) in enumerate(self.GLUT_data[ch].items()):
                    if gu >= indp or gl >= indp:
                        bgid = gi
                        break
                for i, gid in enumerate(self.gate_sequence_ids[ch]):
                    if gid == bgid:
                        inds = i
                        badinds.append(inds)
                        break
            if errg:
                if self.lut_overflow_handling == LUTOverflowHandling.RaiseException:
                    raise LUTOverflowException("GLUT filling exceeded")
                for i, gid in enumerate(self.gate_sequence_ids[ch]):
                    if gid == errg:
                        inds = i
                        badinds.append(inds)
                        break
        _, slexpand = self.lensl(self.slice_list)

        # Bad/failing indices (or addresses) are captured from
        # gate_sequence_ids, which handles loop expansion. The index mapping
        # onto slice_list fails when Loop constructs are used (in jaqal or
        # jaqalpaw) since slice_list is still nested. The self.lensl() function
        # finds the qualified length of the slice list and returns a flattened
        # list of indices. The associated index is either found or approximated
        # by a filter since explicit loop expansion in slice_list poses
        # additional challenges and truncating to the point before the problem
        # loop is hopefully sufficient.
        badindso = list(map(lambda x: len(list(filter(lambda y: y <= x, slexpand))) - 1, badinds))
        if len(badindso) > 0 and min(badindso) < 0:
            raise CircuitCompilerException(f"LUT size exceeded within first loop construct")
        return badindso

    def lensl(self, slicelist, istop=True):
        l = 0
        llist = []
        for sl in slicelist:
            if isinstance(sl, Loop):
                l += self.lensl(sl, istop=False)[0]*sl.repeats
            elif isinstance(sl, list):
                l += self.lensl(sl, istop=False)[0]
            else:
                l += 1
            if istop:
                llist.append(l)
        return l, llist

    def compile(self, channel_mask=None):
        """Compile the circuit, starting from parsing the jaqal file"""
        if self.slice_list is None:
            if self.file is None and self.code_literal is None:
                raise CircuitCompilerException("Need an input file!")
            self.construct_circuit(
                self.file,
                override_dict=self.override_dict,
                pd_override_dict=self.pd_override_dict,
            )
            self.apply_delays(self.delay_settings)
        if self.flattened_slice_list is None:
            self.flattened_slice_list = self.slice_list
        self.extract_gates()
        luts_exceeded = self.generate_lookup_tables()
        if luts_exceeded and self.lut_overflow_handling == LUTOverflowHandling.InterleavedStream:
            self.boardseqdata = self.generate_interleaved_data(channel_mask)
            return None, None, None
        gpres = self.generate_programming_data()
        slice_ind = None
        if gpres:
            if self.lut_overflow_handling in (LUTOverflowHandling.RaiseException, LUTOverflowHandling.StreamDataDirectly):
                raise LUTOverflowException("LUT capacity exceeded")
            slice_ind = min(gpres)
            # Find the longest gate within a reasonable range to inject
            # new programming data in order to prevent FIFO underflows
            if slice_ind > len(self.flattened_slice_list) // 2:
                start_si = len(self.flattened_slice_list) // 2
            else:
                start_si = max(slice_ind - 10, slice_ind // 2)
            dur_list = [self.get_slice_duration(s) for s in reversed(self.flattened_slice_list[start_si:slice_ind])]
            slice_ind -= dur_list.index(max(dur_list))
            cc1 = CircuitCompiler(num_channels=self.channel_num, slice_list=self.flattened_slice_list[:slice_ind])
            self.cclist.append(cc1)
            cc2 = CircuitCompiler(num_channels=self.channel_num, slice_list=self.flattened_slice_list[slice_ind:])
            self.cclist.append(cc2)
        self.compiled = True
        return slice_ind, None, None

    def get_slice_duration(self, s):
        if isinstance(s, GateSlice):
            return s.total_duration()
        else:
            try:
                return sum(map(self.get_slice_duration, s))
            except Exception as e:
                logging.getLogger(__name__).warning(f"Unable to get slice duration: {e}")
                return 0

    def last_packet_pulse_data(self, ch):
        return [PulseData(ch, 3e-7, waittrig=False), PulseData(ch, 3e-7, waittrig=True)]

    def generate_last_packet(self, channel_mask=None):
        packet_data = []
        prog_data = []
        seq_data = []
        for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
            prog_data = []
            seq_data = []
            packet_data = []
            for chnm in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num)):
                if channel_mask is None or (1 << chnm) & channel_mask:
                    for pd in self.last_packet_pulse_data(chnm):
                        packet_data.extend(pd.binarize(bypass=True))
        prog_data.append([])
        timesorted_packets = timesort_bytelist(packet_data)
        seq_data.append(timesorted_packets)
        return prog_data, seq_data

    def bytecode(self, channel_mask=None):
        """Return the bytecode for compiling and running a gate sequence.
        The data is returned in two blocks, programming data and sequence data.
        Each block contains a list of lists, where the structure of each block is

               [[board0 byte list], [board1 byte list], ...]

        for use with multiple boards, and the byte lists are a list of 32 byte (256 bit)
        words that can be concatenated and sent to the device. Sequence data is sent
        separately in case the data needs to be sent with multiple repetitions.

        The channel_mask is an N bit mask that is used to selectively filter out
        data by channel, where 0 prevents the data from being sent and the LSB
        corresponds to channel 0. If channel_mask is None, data is supplied for
        all channels up to self.channel_num"""
        return self.subcircuit_bytecode(channel_mask, combine_subcircuits=True)
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        if not self.compiled:
            try:
                self.compile(channel_mask)
            except LUTOverflowException as e:
                if self.lut_overflow_handling == LUTOverflowHandling.StreamDataDirectly:
                    logging.getLogger(__name__).info("couldn't fit data into LUTs, streaming directly")
                    self.gateTransList = [None]*self.channel_num
                    return [[] for _ in range(self.num_boards)], self.streaming_data_from_luts(channel_mask)
                raise e
        self.final_byte_dict = defaultdict(list)
        self.programming_data = list()
        self.sequence_data = list()
        if len(self.cclist) == 0:
            if self.channel_num > CHANNELS_PER_BOARD:
                for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
                    board_programming_data = list()
                    board_sequence_data = list()
                    for bindata in zip_longest(
                            (self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                        for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                        if (1 << ch) & channel_mask), fillvalue=b''
                    ):
                        board_programming_data.extend(bindata[0])
                    for bindata in zip_longest(
                        *list(
                            self.GSEQ_bin[ch]
                            for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                            if (1 << ch) & channel_mask
                        ), fillvalue=b''
                    ):
                        if bindata:
                            board_sequence_data.extend(bindata)
                    self.programming_data.append(board_programming_data)
                    self.sequence_data.append(board_sequence_data)
            else:
                board_programming_data = list()
                board_sequence_data = list()
                for bindata in zip_longest(
                    (self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                    for ch in range(self.channel_num)
                    if (1 << ch) & channel_mask),
                    fillvalue=b""
                ):
                    board_programming_data.extend(bindata[0])
                for bindata in zip_longest(
                    *list(
                        self.GSEQ_bin[ch]
                        for ch in range(self.channel_num)
                        if (1 << ch) & channel_mask
                    ),
                    fillvalue=b""
                ):
                    if bindata:
                        board_sequence_data.extend(bindata)
                self.programming_data.append(board_programming_data)
                self.sequence_data.append(board_sequence_data)
        else:
            for cc in self.cclist:
                pdat, sdat = cc.bytecode(channel_mask=channel_mask)
                if not self.sequence_data:
                    self.sequence_data = pdat[:]
                    for i, sd in enumerate(sdat):
                        self.sequence_data[i] += sd
                else:
                    for i, pd in enumerate(pdat):
                        self.sequence_data[i] += pd
                    for i, sd in enumerate(sdat):
                        self.sequence_data[i] += sd
            for _ in self.sequence_data:
                self.programming_data.append([])
        return self.consolidate_channel_routing(
                self.programming_data, self.sequence_data, combine_subcircuits=True)

    def consolidate_channel_routing(self, pbytes, sbytes, combine_subcircuits=False):
        """Combine comparable programming/sequence data for n-hot routing.
        Primarily useful for cases where a number of channels are idle or have
        a lot of overlap, but might not have much impact when lots of unique
        data is used."""
        if VERSION == 2:
            finalpbytes = []
            finalsbytes = []
            for n in range(self.num_boards):
                finalpbytes.append(self._consolidate_prog_routing(pbytes[n]))
                if self.boardseqdata is None:
                    if isinstance(sbytes[0][0], list):
                        finalsbytes.append([])
                        for sc in range(len(sbytes[0])):
                            if combine_subcircuits:
                                finalsbytes[n].extend(self._consolidate_seq_routing(sbytes[n][sc]))
                            else:
                                finalsbytes[n].append(self._consolidate_seq_routing(sbytes[n][sc]))
                    else:
                        finalsbytes.append(self._consolidate_seq_routing(sbytes[n]))
                else:
                    if combine_subcircuits:
                        finalsbytes.append(reduce(lambda x,y: x+y, self.boardseqdata[n]))
                    else:
                        finalsbytes = self.boardseqdata
            return finalpbytes, finalsbytes
        return pbytes, sbytes

    def _consolidate_prog_routing(self, blist):
        newbytes = []
        pvd = defaultdict(int)
        for pv in map(bytes_to_int, blist):
            pvm = pv & ((1<<DMA_MUX_LSB)-1)
            pvr = pv >> DMA_MUX_LSB
            pvd[pvm] |= pvr
        for pvk,pvv in pvd.items():
            newbytes.append(int_to_bytes(pvk|(pvv<<DMA_MUX_LSB)))
        return newbytes

    def _consolidate_seq_routing(self, blist):
        return blist
        newsbytes = []
        pvd = defaultdict(int)
        for chdat in group_adjacent(blist,CHANNELS_PER_BOARD):
            pvd = defaultdict(int)
            for pv in map(bytes_to_int, chdat):
                pvm = pv & ((1<<DMA_MUX_LSB)-1)
                pvr = pv >> DMA_MUX_LSB
                pvd[pvm] |= pvr
            for pvk,pvv in pvd.items():
                newsbytes.append(int_to_bytes(pvk|(pvv<<DMA_MUX_LSB)))
        return newsbytes

    def get_prepare_all_indices(self):
        self.prepare_all_hashes = dict()
        self.prepare_all_gids = dict()
        prepare_all_prefix = "gate_"
        if not hasattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name):
            prepare_all_prefix = "macro_"
            if not hasattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name):
                raise CircuitCompilerException(
                f"Pulse definition has no gate named gate_{self.initialize_gate_name}"
                )
        if prepare_all_prefix == "macro_":
            gate_data = getattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name)(
                self.channel_num
            )[0]
        else:
            gate_data = getattr(self.pulse_definition, prepare_all_prefix + self.initialize_gate_name)(
                self.channel_num
            )
        gslice = populate_gate_slice(gate_data, self.channel_num)
        self.apply_delays(self.delay_settings, circ_main=gslice)
        for ch, gsdata in gslice.channel_data.items():
            prep_hash = hash(tuple(gsdata))
            self.prepare_all_hashes[ch] = prep_hash
            inverted_gid_hashes = {
                v: k for k, v in self.ordered_gate_identifiers[ch].items()
            }
            gid = inverted_gid_hashes.get(prep_hash, None)
            if gid is None:
                raise CircuitCompilerException(
                    f"Unable to find hash for {self.initialize_gate_name}"
                )
            self.prepare_all_gids[ch] = gid
        if ENABLE_MLUT_PACKING:
            for ch in range(self.channel_num):
                if self.gateTransList[ch] is not None:
                    self.prepare_all_gids[ch] = self.gateTransList[ch][self.prepare_all_gids[ch]][0]
        return gslice

    def generate_gate_sequence_subcircuits_inds_only(self):
        """Generate gate sequence ids broken up by subcircuit"""
        self.get_prepare_all_indices()
        partial_gs_ids = dict()
        partial_GSEQ_bin = dict()
        for ch, gidlist in self.gate_sequence_ids.items():
            partial_gs_ids[ch] = get_subcircuits(
                self.prepare_all_gids[ch], gidlist
            )
        return partial_gs_ids

    def generate_gate_sequence_from_index(self, ind):
        """Generate gate sequence bytes from a given prepare_all index"""
        partial_gs_ids = self.generate_gate_sequence_subcircuits_inds_only()
        partial_GSEQ_bin = dict()
        for ch in partial_gs_ids:
            partial_GSEQ_bin[ch] = gate_sequence_bytes(partial_gs_ids[ch], ch)
        return partial_GSEQ_bin

    def generate_gate_sequence_subcircuits(self):
        partial_gs_ids = self.generate_gate_sequence_subcircuits_inds_only()
        partial_GSEQ_bin = dict()
        for ch in partial_gs_ids:
            partial_GSEQ_bin[ch] = list(map(lambda x: gate_sequence_bytes(x,ch),
                                            partial_gs_ids[ch]))
        return partial_GSEQ_bin

    def partial_sequence_bytecode(self, channel_mask=None, starting_index=0):
        """Reduced form of bytecode function that only generates the sequence data from a given index"""
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        if not self.compiled:
            self.compile(channel_mask)
        self.final_byte_dict = defaultdict(list)
        programming_data = list()
        sequence_data = list()
        partial_GSEQ_bin = self.generate_gate_sequence_from_index(starting_index)
        if self.channel_num > CHANNELS_PER_BOARD:
            for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
                board_programming_data = list()
                board_sequence_data = list()
                for bindata in zip_longest(
                    *list(
                        partial_GSEQ_bin[ch]
                        for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                        if (1 << ch) & channel_mask
                    ),
                    fillvalue=b'',
                ):
                    if bindata:
                        board_sequence_data.extend(bindata)
                programming_data.append(board_programming_data)
                sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            for bindata in zip_longest(
                *list(
                    partial_GSEQ_bin[ch]
                    for ch in range(self.channel_num)
                    if (1 << ch) & channel_mask
                ),
                fillvalue=b'',
            ):
                if bindata:
                    board_sequence_data.extend(bindata)
            programming_data.append(board_programming_data)
            sequence_data.append(board_sequence_data)
        return programming_data, sequence_data

    def subcircuit_bytecode(self, channel_mask=None, combine_subcircuits=False):
        """Return the bytecode for compiling and running a gate sequence.
        The data is returned in two blocks, programming data and sequence data.
        Each block contains a list of lists, where the structure of each block is

               [[board0 byte list], [board1 byte list], ...]

        for use with multiple boards, and the byte lists are a list of 32 byte (256 bit)
        words that can be concatenated and sent to the device. Sequence data is sent
        separately in case the data needs to be sent with multiple repetitions.

        The channel_mask is an N bit mask that is used to selectively filter out
        data by channel, where 0 prevents the data from being sent and the LSB
        corresponds to channel 0. If channel_mask is None, data is supplied for
        all channels up to self.channel_num"""
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        if not self.compiled:
            try:
                self.compile(channel_mask)
            except LUTOverflowException as e:
                if self.lut_overflow_handling == LUTOverflowHandling.StreamDataDirectly:
                    logging.getLogger(__name__).info("couldn't fit data into LUTs, streaming directly")
                    self.gateTransList = [None]*self.channel_num
                    if combine_subcircuits:
                        return [[] for _ in range(self.num_boards)], self.streaming_data_from_luts(channel_mask)
                    return [[] for _ in range(self.num_boards)], self.subcircuit_streaming_data_from_luts(channel_mask)
                raise e
        self.final_byte_dict = defaultdict(list)
        self.programming_data = list()
        self.sequence_data = list()
        subcircuit_GSEQ_bin = self.generate_gate_sequence_subcircuits()
        if self.channel_num > CHANNELS_PER_BOARD:
            for bbind in range(0, self.channel_num, CHANNELS_PER_BOARD):
                board_programming_data = list()
                board_sequence_data = list()
                for bindata in zip_longest(
                        (self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                        for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                        if (1 << ch) & channel_mask),
                        fillvalue=b'',
                ):
                    board_programming_data.extend(bindata[0])
                for subcircuit_index in range(len(subcircuit_GSEQ_bin[0])):
                    subcirc_index_board_data = []
                    for bindata in zip_longest(
                            *list(
                                subcircuit_GSEQ_bin[ch][subcircuit_index]
                                for ch in range(bbind, min(bbind + CHANNELS_PER_BOARD, self.channel_num))
                                if (1 << ch) & channel_mask
                            ),
                            fillvalue=b'',
                    ):
                        if bindata:
                            subcirc_index_board_data.extend(bindata)
                    board_sequence_data.append(subcirc_index_board_data)
                self.programming_data.append(board_programming_data)
                self.sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            for bindata in zip_longest(
                    (self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                    for ch in range(self.channel_num)
                    if (1 << ch) & channel_mask),
                    fillvalue=b'',
            ):
                board_programming_data.extend(bindata[0])
            for subcircuit_index in range(len(subcircuit_GSEQ_bin[0])):
                subcirc_index_board_data = []
                for bindata in zip_longest(
                        *list(
                            subcircuit_GSEQ_bin[ch][subcircuit_index]
                            for ch in range(self.channel_num)
                            if (1 << ch) & channel_mask
                        ),
                        fillvalue=b'',
                ):
                    if bindata:
                        subcirc_index_board_data.extend(bindata)
                board_sequence_data.append(subcirc_index_board_data)
            self.programming_data.append(board_programming_data)
            self.sequence_data.append(board_sequence_data)
        return self.consolidate_channel_routing(
                self.programming_data, self.sequence_data, combine_subcircuits=combine_subcircuits)


def revindex(ll,val):
    """Like list.index(val), but find the last index"""
    for i,l in enumerate(reversed(ll)):
        if l == val:
            return len(ll)-i-1
    return None


def make_glut_tree(glutremap):
    """Each gate comprises N MLUT data words that contain routing information
       and PLUT addresses. Make a tree that the corresponds to the ordered MLUT
       data where the root node is the first element."""
    gluttree = tree()
    for gid, maddrs in glutremap.items():
        gt = gluttree
        for i in maddrs:
            gt = gt[i]
        gt[-1] = gid
    return gluttree


def find_longest_root(dd,depth=0,vl=None):
    """Walk the tree and look for nodes that have the largest product of
    depth*breadth. In other words, use the tree to identify overlap in data
    which affects the largest MLUT filling while maintaining order and repeated
    entries (as opposed to a set-based approach). In this case we target
    overlap for the beginning of each sequence."""
    maxvl = []
    if vl is None:
        vl = []
    maxv = depth*(len(dd.keys())-1)
    maxk = vl
    for k,v in dd.items():
        if k > -1:
            newd, kl = find_longest_root(dd[k],depth+1, vl+[k])
            if newd > maxv:
                maxv = newd
                maxk = kl
    return maxv, maxk


def get_remap_candidates(dd, vl=None, limit=None):
    """Given a subtree, return all valid children. Here we limit the gate id to
    prevent optimization on generated gatelets since this starts to get
    annoying with a recursive remap of gate ids."""
    if vl is None:
        vl = []
    for k,v in dd.items():
        if k != -1:
            yield from get_remap_candidates(dd[k], (*vl,k), limit=limit)
        else:
            if vl:
                if limit is not None and v>limit:
                    continue
                yield (vl, v)


def optimluts(glut, glutmap=None, limit=None):
    """Generate tree representations for gates run forward and reverse (the
    latter is used to target gates that end with common data, but the rest of
    the code handles ordering in one direction so we can handle the reverse
    case at this level). Find the optical target gatelet and separate it from
    the relevant gates."""
    if glutmap is None:
        glutmap = {i:(i,) for i in glut.keys()}
    glutr = {k:tuple(reversed(v)) for k,v in glut.items()}
    tf = make_glut_tree(glut)
    tfr = make_glut_tree(glutr)
    maxv, maxk = find_longest_root(tf)
    maxvr, maxkr = find_longest_root(tfr)
    if maxvr > maxv:
        return single_pass_distillation(glut, glutmap, tfr, maxkr, rev=True, limit=limit), maxvr
    else:
        return single_pass_distillation(glut, glutmap, tf, maxk, rev=False, limit=limit), maxv


def single_pass_distillation(glut, glutmap, tf, maxk, rev=False, limit=None):
    """Distill gates into gatelets, augment the gate table and generate a
       translation dict for expanding gates into multiple gatelets. Order is
       preserved and new gatelets are added to the end of the table."""
    ttf = tf
    for i in maxk:
        ttf = ttf[i]
    # check if gatelet exists (this might happen if it shows up multiple times
    # in a gate, or at the beginning of one gate and the end of another (or the
    # same) gate
    target_tup = tuple(reversed(maxk)) if rev else tuple(maxk)
    if target_tup in glut.values():
        newglutid = list(glut.values()).index(target_tup)
    else:
        newglutid = len(glut)
        glutmap[newglutid] = [newglutid]
    rmapl = list(get_remap_candidates(ttf, limit=limit))
    if len(rmapl) < 1:  # limit distillation below a certain threshold
        return glut, glutmap
    for (gl,ngid) in get_remap_candidates(ttf, limit=limit):
        # This is a bit annoying, but because we keep augmenting the
        # translation for gates to gatelets in glutmap, we have to insert
        # gatelets in a reverse middle-out order. In other words, given a
        # gate G that gets separated into shared gatelets named A,B,C,...,
        # original gate data being broken down into G', G'', G''', ... with
        # each iteration, we need to sort the gatelets in the following way
        #
        #       G ->       (A, G')             (best match in FORWARD tree)
        #         ->       (A, G'', B)         (best match in REVERSE tree)
        #         ->    (A, C, G''', B)        (best match in FORWARD tree)
        #         -> (A, C, D, G'''', B)       (best match in FORWARD tree)
        #         -> (A, C, D, G''''', E, B)   (best match in REVERSE tree)
        #
        # Might want to come up with a cleaner way of dealing with this but
        # this is expected to be a fairly uncommon occurrence, but one that
        # needs to be handled correctly. Otherwise we could get rid of the
        # newv/satisfied loop in the following two cases
        if rev:
            target = next(filter(lambda x: x<=limit, reversed(glutmap[ngid])))
            idx = revindex(glutmap[ngid], target)
            glutmap[ngid][idx+1:idx+1] = [newglutid]
            glut[ngid] = tuple(reversed(gl))
        else:
            target = next(filter(lambda x: x<=limit, glutmap[ngid]))
            idx = glutmap[ngid].index(target)
            glutmap[ngid][idx:idx] = [newglutid]
            glut[ngid] = tuple(gl)
    if rev:
        glut[newglutid] = tuple(reversed(maxk))
    else:
        glut[newglutid] = tuple(maxk)
    return glut, glutmap


def optim_reasonable(glut, bound=None):
    """Determine if continued optimization is reasonable based on some bound.
       Otherwise default to reduction by 10% which isn't much but it's some
       kind of bound if one isn't explicitly set. In this case we can compare
       the number of unique and non-unique entries and determine if it's
       possible to reduce redundancies below the bound if the number of unique
       entries is less than the bound."""
    mlutflat=[mlutdat for mlutl in glut.values() for mlutdat in mlutl]
    mlutc = set(mlutflat)
    if len(glut.values()) >= (1<<GLUTW):
        return True, len(mlutflat)
    if bound:
        if len(mlutc) < bound and len(mlutflat) > bound:
            return True, len(mlutflat)
        return False, len(mlutflat)
    return len(mlutc)/len(mlutflat) < 0.90, len(mlutflat)


def distill_gatelets(nglut, bound=None, maxiter=None):
    """Top-level call for distilling gatelets. Loop over optimluts if we an
    reasonably hit our optimization target. If limitations in ordering prevent
    optimization from the endpoints of the MLUT sequence then break out of the
    loop."""
    glutmap = {i:[i] for i in range(len(nglut))}
    opt = True
    lastcount = 0
    limit = max(nglut)
    opt, lastcount = optim_reasonable(nglut, bound)
    lastcountn = lastcount
    itr = 0
    if maxiter is None:
        maxiter = 1e6
    while opt:
        if itr >= maxiter:
            break
        itr += 1
        logging.getLogger(__name__).debug(
            f"Optimizing gatelets..."
            )
        (nglut, glutmap), maxvn = optimluts(nglut.copy(), glutmap, limit=limit)
        opt, lastcountn = optim_reasonable(nglut, bound)
        if lastcountn == lastcount:
            break
        lastcount = lastcountn
    return nglut, glutmap, lastcountn


def verify_distillation(oglut,nglut,glutmap):
    _nglut = {}
    for ogid in oglut:
        gl = []
        for gid in glutmap[ogid]:
            gl.extend(nglut[gid])
        _nglut[ogid] = tuple(gl)
        assert oglut[ogid] == _nglut[ogid]


# ######################################################## #
# ------- Temporary jaqal Preprocessing Functions -------- #
# ######################################################## #


def split_lets(instr):
    """check lines for the existence of 'let' if it doesn't appear
    after a comment and return a tuple of the name and value strings"""
    inlist = list(filter(lambda x: len(x) > 0, instr.partition("//")[0].split(" ")))
    if inlist and inlist[0].strip() == "let":
        return (inlist[1].strip(), inlist[2].strip())
    return None


def split_usepulses(instr):
    """Temporary usepulses parser for specifying pulse definition class.
    This is a placeholder until imports are resolved, and specifically
    looks for a commented line to avoid problems with parsing."""
    pd = instr.partition("//")[2].partition("usepulses")[2].strip()
    return pd or None


def float_or_int(val):
    """Try to convert to an int, otherwise return a float"""
    try:
        return int(val)
    except ValueError as e:
        return float(val)


def get_jaqal_lets(filename, return_defaults=False):
    """Find all valid 'let' statements for an input jaqal file
    and return their names (return_defaults == False) or a list
    of tuples with the names and values (return_defaults == True)"""
    jp = Path(filename)
    output = list(
        filter(lambda x: x is not None, map(split_lets, jp.read_text().splitlines()))
    )
    if return_defaults:
        return {name: float_or_int(val) for (name, val) in output}
    return list(list(zip(*output))[0])


def get_gate_pulse_name(filename):
    """Get the gate pulse class name specified by a
    '//usepulses MyGatePulseClass' call in the jaqal file"""
    jp = Path(filename)
    output = list(
        filter(
            lambda x: x is not None, map(split_usepulses, jp.read_text().splitlines())
        )
    )
    if output:
        return output[0]
    return None


def get_tail_from_index(elem, gidlist, ind):
    from itertools import compress, count

    gid_ind = list(compress(count(), map(lambda x: x == elem, gidlist)))[ind]
    return gidlist[gid_ind:]


def get_subcircuits(elem, gidlist):
    from itertools import compress, count, zip_longest
    gid_start_indices = list(compress(count(), map(lambda x: x == elem, gidlist)))
    gid_end_indices = gid_start_indices[1:]
    subcircuit_gids = []
    for (gstart, gend) in zip_longest(gid_start_indices, gid_end_indices):
        subcircuit_gids.append(gidlist[gstart:gend])
    return subcircuit_gids

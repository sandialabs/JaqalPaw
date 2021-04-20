from pathlib import Path
from collections import defaultdict
from itertools import zip_longest

from jaqalpaw.ir.circuit_constructor import CircuitConstructor
from jaqalpaw.ir.gate_slice import GateSlice
from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.bytecode.lut_programming import (
    program_PLUT,
    program_SLUT,
    program_GLUT,
    gate_sequence_bytes,
)
from .time_ordering import timesort_bytelist
from jaqalpaw.utilities.datatypes import Loop, to_clock_cycles, Branch, Case
from jaqalpaw.utilities.exceptions import CircuitCompilerException
from jaqalpaw.utilities.parameters import CLKFREQ
from jaqalpaw.bytecode.encoding_parameters import (
    ANCILLA_COMPILER_TAG_BIT,
    ANCILLA_STATE_LSB,
)
from ..ir.circuit_constructor_visitor import populate_gate_slice

flatten = lambda x: [y for l in x for y in l]

# ######################################################## #
# ---------- Convert GateSlice IR to Bytecode ------------ #
# ######################################################## #


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
    ):
        super().__init__(num_channels, pulse_definition)
        self.file = file
        self.code_literal = code_literal
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
        self.import_gate_pulses()

    def set_global_delay(self, global_delay=None):
        if global_delay is None:
            self.delay_settings = None
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

    def recursive_append(self, slices, appendto):
        """Walk nested lists but don't expand loops"""
        for s in slices:
            if isinstance(s, Loop):
                for _ in range(s.repeats):
                    self.recursive_append(s, appendto)
            elif isinstance(s, list):
                self.recursive_append(s, appendto)
            else:
                appendto.append(s)

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
        if delay_settings is None:
            return
        if circ_main is None:
            circ_main = GateSlice(num_channels=self.channel_num)
            self.recursive_append(self.slice_list, circ_main)
        for ch, pd_list in circ_main.channel_data.items():
            for pd in pd_list:
                if pd.waittrig:
                    pd.delay = delay_settings[ch]

    def streaming_data(self, channels=None):
        """Generate the binary data for direct streaming (bypass mode)"""
        self.binarize_circuit(bypass=True)
        if channels is None:
            channels = list(range(self.channel_num))
        bytelist = []
        for ch in channels:
            bytelist.extend(self.binary_data[ch])
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
                    reps=reps,
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
                                    | (1 << ANCILLA_COMPILER_TAG_BIT)
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
        for ch in range(self.channel_num):
            gid = 0
            addr = 0
            for k, v in sorted(self.ordered_gate_identifiers[ch].items()):
                gate_start_addr = addr
                for pd in self.unique_gates[ch][v]:
                    pd_bin = pd.binarize()
                    for pdb in pd_bin:
                        if pdb not in self.PLUT_data[ch]:
                            self.PLUT_data[ch].append(pdb)
                        self.MMAP_data[ch][addr] = self.PLUT_data[ch].index(pdb)
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
                            (startind + state + gind) | (1 << ANCILLA_COMPILER_TAG_BIT)
                        ] = self.GLUT_data[ch][subgid]
                        gind += 1
                startind += maxlen

    def generate_programming_data(self):
        """Convert the LUT programming IR representations to bytecode"""
        self.PLUT_bin = defaultdict(list)
        self.MMAP_bin = defaultdict(list)
        self.GLUT_bin = defaultdict(list)
        self.GSEQ_bin = defaultdict(list)
        for ch in range(self.channel_num):
            self.PLUT_bin[ch] = program_PLUT(
                {v: i for i, v in enumerate(self.PLUT_data[ch])}, ch
            )
            self.MMAP_bin[ch] = program_SLUT(self.MMAP_data[ch], ch)
            self.GLUT_bin[ch] = program_GLUT(self.GLUT_data[ch], ch)
            self.GSEQ_bin[ch] = gate_sequence_bytes(self.gate_sequence_ids[ch], ch)

    def compile(self):
        """Compile the circuit, starting from parsing the jaqal file"""
        if self.file is None and self.code_literal is None:
            raise CircuitCompilerException("Need an input file!")
        self.construct_circuit(
            self.file,
            override_dict=self.override_dict,
            pd_override_dict=self.pd_override_dict,
        )
        self.apply_delays(self.delay_settings)
        self.extract_gates()
        self.generate_lookup_tables()
        self.generate_programming_data()
        self.compiled = True

    def last_packet_pulse_data(self, ch):
        return [PulseData(ch, 3e-7, waittrig=False), PulseData(ch, 3e-7, waittrig=True)]

    def generate_last_packet(self, channel_mask=None):
        packet_data = []
        prog_data = []
        seq_data = []
        for bbind in range(0, self.channel_num, 8):
            prog_data = []
            seq_data = []
            packet_data = []
            for chnm in range(bbind, min(bbind + 8, self.channel_num)):
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
        if not self.compiled:
            self.compile()
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        self.final_byte_dict = defaultdict(list)
        self.programming_data = list()
        self.sequence_data = list()
        if self.channel_num > 8:
            for bbind in range(0, self.channel_num, 8):
                board_programming_data = list()
                board_sequence_data = list()
                for bindata in zip_longest(
                    self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                    for ch in range(bbind, min(bbind + 8, self.channel_num))
                    if (1 << ch) & channel_mask
                ):
                    board_programming_data.extend(bindata[0])
                for bindata in zip_longest(
                    *list(
                        self.GSEQ_bin[ch]
                        for ch in range(bbind, min(bbind + 8, self.channel_num))
                        if (1 << ch) & channel_mask
                    )
                ):
                    if bindata:
                        board_sequence_data.extend(bindata)
                self.programming_data.append(board_programming_data)
                self.sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            for bindata in zip_longest(
                self.GLUT_bin[ch] + self.MMAP_bin[ch] + self.PLUT_bin[ch]
                for ch in range(self.channel_num)
                if (1 << ch) & channel_mask
            ):
                board_programming_data.extend(bindata[0])
            for bindata in zip_longest(
                *list(
                    self.GSEQ_bin[ch]
                    for ch in range(self.channel_num)
                    if (1 << ch) & channel_mask
                )
            ):
                if bindata:
                    board_sequence_data.extend(bindata)
            self.programming_data.append(board_programming_data)
            self.sequence_data.append(board_sequence_data)
        return self.programming_data, self.sequence_data

    def get_prepare_all_indices(self):
        self.prepare_all_hashes = dict()
        self.prepare_all_gids = dict()
        if not hasattr(self.pulse_definition, "gate_" + self.initialize_gate_name):
            raise CircuitCompilerException(
                f"Pulse definition has no gate named gate_{self.initialize_gate_name}"
            )
        gate_data = getattr(self.pulse_definition, "gate_" + self.initialize_gate_name)(
            self.channel_num
        )
        gslice = populate_gate_slice(gate_data, self.channel_num)
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
        return gslice

    def generate_gate_sequence_from_index(self, ind):
        self.get_prepare_all_indices()
        partial_gs_ids = dict()
        partial_GSEQ_bin = dict()
        for ch, gidlist in self.gate_sequence_ids.items():
            partial_gs_ids[ch] = get_tail_from_index(
                self.prepare_all_gids[ch], gidlist, ind
            )
            partial_GSEQ_bin[ch] = gate_sequence_bytes(partial_gs_ids[ch], ch)
        return partial_GSEQ_bin

    def partial_sequence_bytecode(self, channel_mask=None, starting_index=0):
        """Reduced form of bytecode function that only generates the sequence data from a given index"""
        if not self.compiled:
            self.compile()
        if channel_mask is None:
            channel_mask = (1 << self.channel_num) - 1
        self.final_byte_dict = defaultdict(list)
        programming_data = list()
        sequence_data = list()
        partial_GSEQ_bin = self.generate_gate_sequence_from_index(starting_index)
        if self.channel_num > 8:
            for bbind in range(0, self.channel_num, 8):
                board_programming_data = list()
                board_sequence_data = list()
                # for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
                # for ch in range(bbind, min(bbind+8, self.channel_num))
                # if (1 << ch) & channel_mask):
                # board_programming_data.extend(bindata[0])
                for bindata in zip_longest(
                    *list(
                        partial_GSEQ_bin[ch]
                        for ch in range(bbind, min(bbind + 8, self.channel_num))
                        if (1 << ch) & channel_mask
                    )
                ):
                    if bindata:
                        board_sequence_data.extend(bindata)
                programming_data.append(board_programming_data)
                sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            # for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
            # for ch in range(self.channel_num)
            # if (1 << ch) & channel_mask):
            # board_programming_data.extend(bindata[0])
            for bindata in zip_longest(
                *list(
                    partial_GSEQ_bin[ch]
                    for ch in range(self.channel_num)
                    if (1 << ch) & channel_mask
                )
            ):
                if bindata:
                    board_sequence_data.extend(bindata)
            programming_data.append(board_programming_data)
            sequence_data.append(board_sequence_data)
        return programming_data, sequence_data


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

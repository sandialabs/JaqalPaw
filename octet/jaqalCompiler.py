from octet.intermediateRepresentations import GateSlice, PulseData, clock_cycles, CircuitCompilerException
from pathlib import Path
from collections import defaultdict
from octet.LUTProgramming import programGLUT, programPLUT, programSLUT, gateSequenceBytes
from itertools import zip_longest
from octet.encodingParameters import MODTYPE_LSB, DMA_MUX_OFFSET, ENDIANNESS
#from jaqal.jaqal.interface import Interface, MemoizedInterface
from jaqalpaq.parser import parse_jaqal_string
from jaqalpaq.core.algorithm import expand_macros, fill_in_let
from jaqalpaq.core.algorithm.visitor import Visitor
import time
from functools import lru_cache
import runpy
from itertools import zip_longest

flatten = lambda x: [y for l in x for y in l]

# ######################################################## #
# --------------- Time Ordering Functions ---------------- #
# ######################################################## #


class TimeStampedWord:
    def __init__(self, duration, word, start_time=0, mod_type=None, chan=None):
        self.duration = duration
        self.word = word
        self.start_time = start_time
        self.mod_type = mod_type
        self.chan = chan

    @property
    def end_time(self):
        return self.duration + self.start_time

    def __add__(self, other):
        if not isinstance(other, TimeStampedWord):
            raise Exception(f"Can't add to object of type {type(other)}")
        elif self.chan != other.chan:
            raise Exception(f"Can't add time to other channel's data")
        else:
            return TimeStampedWord(self.duration, self.word, start_time=other.end_time)

    def __repr__(self):
        return f"type: {self.mod_type} start: {self.start_time}"


def mapFromBytes(d, bytenum=5):
    return [int.from_bytes(d[n*bytenum:n*bytenum+bytenum], byteorder=ENDIANNESS, signed=True) for n in range(bytenum)]


def decode_word(word):
    """Extract the channel, modulation type and duration from a data word"""
    data = int.from_bytes(word, byteorder='little', signed=False)
    mod_type = (data >> MODTYPE_LSB) & 0b111
    channel = (data >> DMA_MUX_OFFSET) & 0b111
    U0, U1, U2, U3, dur = mapFromBytes(word)
    return channel, mod_type, dur


def generate_time_stamped_data(bytelist):
    """Convert a list of 256 bit words to TimeStampedWord objects that
       calculates the start time for each word in the sequence"""
    parameter_dd = defaultdict(lambda: defaultdict(list))
    full_pb_list = []
    for pb in bytelist:
        chan, mod_type, dur = decode_word(pb)
        start_time = 0
        if len(parameter_dd[chan][mod_type]):
            start_time = parameter_dd[chan][mod_type][-1].end_time
        parameter_dd[chan][mod_type].append(TimeStampedWord(dur, pb, start_time=start_time, mod_type=mod_type, chan=chan))
        full_pb_list.append(TimeStampedWord(dur, pb, start_time=start_time, mod_type=mod_type, chan=chan))
    return full_pb_list


def timesort_bytelist(bytelist):
    """Sort a list of raw data words by start time, then by channel and modulation type"""
    full_pb_list = generate_time_stamped_data(bytelist)
    # sorted_pb_list = list(sorted(full_pb_list, key=lambda el: (el.start_time << 6) | (el.chan << 3) | el.mod_type))
    sorted_pb_list = list(sorted(full_pb_list, key=lambda el: (el.start_time << 6) | (el.mod_type << 3) | el.chan))
    wordlist = []
    for spb in sorted_pb_list:
        wordlist.append(spb.word)
    return wordlist


# ######################################################## #
# ------ Convert jaqal AST to GateSlice IR Layer --------- #
# ######################################################## #

class Loop(list):
    def __init__(self, *args, repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeats = repeats


class CircuitConstructor:
    """Walks the jaqal AST and constructs a list of GateSlice
       objects padding gaps with NOPs and ensuring no collisions"""

    def __init__(self, channel_num, pulse_definition):
        self.CHANNEL_NUM = channel_num
        self.slice_list = []
        self.pulse_definition = pulse_definition
        self.exported_constants = None
        self.reg_list = None
        self.gate_pulse_info = None
        # The circuit before any transformations have been done
        self.base_circuit = None
        # The circuit after certain transformations (such as
        # overriding let variables) has occurred.
        self.circuit = None

    def get_dependencies(self):
        ast = self.generate_ast()
        return get_let_constants(ast), self.gate_pulse_info

    def import_gate_pulses(self):
        if self.gate_pulse_info is None:
            self.get_dependencies()
        if self.gate_pulse_info is None:
            raise CircuitCompilerException("No gate pulse file specified!")
        gp_path = Path(self.file).parent
        jaqal_lets, self.gate_pulse_info = self.get_dependencies()
        gp_name = self.gate_pulse_info[-1]  # jaqal token returns a list of imports (split at '.'), last one is class name
        for p in self.gate_pulse_info[:-1]:
            gp_path /= p  # construct path object from usepulses call
        gp_path = gp_path.with_suffix('.py')
        if gp_path.exists():
            self.gate_pulse_file_path = str(gp_path)
        else:
            raise CircuitCompilerException(f"Can't find path {str(gp_path)}")
        pd_import = runpy.run_path(gp_path, init_globals={'PulseData': PulseData})
        self.pulse_definition = pd_import[gp_name]()
        return self.pulse_definition

    def generate_ast(self, file=None, override_dict=None):
        if self.base_circuit is None:
            if self.file is None:
                text = self.code_literal
            else:
                text = Path(self.file).read_text()
            circuit, extra = parse_jaqal_string(text, autoload_pulses=False, return_usepulses=True)
            usepulses = extra['usepulses']
            self.base_circuit = expand_macros(circuit)
            self.gate_pulse_info = list(usepulses.keys())[0]

        if override_dict is not None:
            self.circuit = fill_in_let(self.circuit, override_dict)
        else:
            self.circuit = self.base_circuit

        return self.circuit

    def construct_circuit(self, file, override_dict=None):
        """Generate full circuit from jaqal file. Circuit is in the form of
        PulseData objects."""
        ast = self.generate_ast(file, override_dict=override_dict)
        self.slice_list = convert_circuit_to_gateslices(
            self.pulse_definition, ast, self.CHANNEL_NUM)


def get_let_constants(ast):
    """Return a list mapping let constant names to their numeric values."""
    return {name: normalize_number(const.value)
            for name, const in ast.constants.items()}


def normalize_number(value):
    """Return an int if the value is an integer (regardless of whether it
    is represented as a float), or a float otherwise."""
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        if int(value) == value:
            return int(value)
    else:
        raise TypeError("Can only normalize ints and floats")


def convert_circuit_to_gateslices(pulse_definition, circuit, num_channels):
    """Convert a Circuit into a list of GateSlice objects."""
    visitor = CircuitConstructorVisitor(pulse_definition, num_channels)
    return visitor.visit(circuit)


class CircuitConstructorVisitor(Visitor):
    """Convert a Circuit into a list of GateSlice objects."""

    def __init__(self, pulse_definition, num_channels):
        super().__init__()
        self.pulse_definition = pulse_definition
        self.num_channels = num_channels

    def visit_Circuit(self, circuit, **kwargs):
        slice_list = self.visit(circuit.body)

        for slc in slice_list:
            slc.make_durations_equal()

        return slice_list

    def visit_BlockStatement(self, block, **kwargs):
        """Return a list of GateSlice's or Loop's from this block."""
        slice_list = []
        if block.parallel:
            for stmt in block.statements:
                stmt_slices = self.visit(stmt)
                slice_list = merge_slice_lists(slice_list, stmt_slices)
        else:
            for stmt in block.statements:
                slice_list.extend(self.visit(stmt, equalize_gate_durations=False))

        return slice_list

    def visit_GateStatement(self, gate, equalize_gate_durations=True):
        """Create a list of a single GateSlice representing this gate."""
        gslice = GateSlice(num_channels=self.num_channels)
        if not hasattr(self.pulse_definition, 'gate_'+gate.name):
            raise CircuitCompilerException(f"Gate {gate.name} not found")
        if is_total_gate(gate.name):
            args = [self.num_channels]
            if len(gate.parameters) > 0:
                raise CircuitCompilerException(f"gate {gate.name} cannot have parameters")
        else:
            args = [self.visit(garg) for garg in gate.parameters.values()]
        gate_data = get_gate_data(self.pulse_definition, gate.name, args)
        if gate_data is not None:
            for pd in gate_data:
                if pd.dur > 3:
                    gslice.channel_data[pd.channel].append(pd)
        return [gslice]

    def visit_int(self, obj, **kwargs):
        """Integer gate arguments remain unchanged."""
        return obj

    def visit_float(self, obj, **kwargs):
        """Float gate arguments remain unchanged."""
        return normalize_number(obj)

    def visit_NamedQubit(self, qubit, **kwargs):
        """Return the index of this qubit in its register. The gate will know
        by its position that this is a qubit index and not an integer."""
        _, index = qubit.resolve_qubit()
        return index


def is_total_gate(gate_name):
    """Return if this gate uses all available qubits without explicitly
    mentioning them as arguments."""
    return gate_name in ['prepare_all', 'measure_all']


def gate_pulse_exists(pulse_definition, gate_name):
    """Return whether the given pulse definition object has the given
    gate."""
    return hasattr(pulse_definition, make_gate_function_name(gate_name))


def get_gate_data(pulse_definition, gate_name, args):
    """Evaluate and return the gate data for a gate with the given
    arguments. The gate is looked up in pulse_definition then evaluated
    with args, which must be converted to numbers."""
    if not all(isinstance(arg, (int, float)) for arg in args):
        # This is a programming error that should be fixed
        raise CircuitCompilerException(f"Bad arg type in {args}")
    pulse_gate = getattr(pulse_definition, make_gate_function_name(gate_name))
    return pulse_gate(*args)


def make_gate_function_name(gate_name):
    """Make the name of the function to look up a gate by in the pulse
    definition object."""
    return f"gate_{gate_name}"


def merge_slice_lists(dst_list, src_list):
    """Take two lists of GateSlice objects and merge them. Overwrites
    dst_list."""

    return [merge_slices(dst, src)
            for dst, src in zip_longest(dst_list, src_list)]


def merge_slices(dst, src):
    """Merge two GateSlice objects. Handles the case where one or the
    other is None. This will always return one of the arguments,
    possibly modified, so the arguments must not be used afterwards.

    """

    if dst is None:
        assert src is not None
        return src
    elif src is None:
        assert dst is not None
        return dst
    else:
        dst.merge(src)
        return dst


def iter_gate_parameters(gate):
    """Iterate over gate parameters in order."""
    parameter_types = gate.gate_def.parameters
    for param in parameter_types:
        yield gate.parameters[param.name]


# ######################################################## #
# ---------- Convert GateSlice IR to Bytecode ------------ #
# ######################################################## #


class CircuitCompiler(CircuitConstructor):
    """Compiles the bytecode to be uploaded to the Octet from
       the intermediate representation layer of GateSlice objects"""

    def __init__(self, file=None, num_channels=8, override_dict=None,
                 pulse_definition=None, global_delay=None, code_literal=None):
        super().__init__(num_channels, pulse_definition)
        self.CHANNEL_NUM = num_channels
        self.file = file
        self.code_literal = code_literal
        self.binary_data = defaultdict(list)
        self.unique_gates = defaultdict(dict)
        self.unique_gate_identifiers = defaultdict(lambda: defaultdict(int))
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
        self.compiled = False
        self.delay_settings = None
        self.set_global_delay(global_delay)
        self.initialize_gate_name = 'prepare_all'
        self.import_gate_pulses()

    def set_global_delay(self, global_delay=None):
        if global_delay is None:
            self.delay_settings = None
        else:
            default_delay = 0
            if global_delay < 0:
                default_delay = -global_delay
                global_delay = 0
            self.delay_settings = defaultdict(lambda: clock_cycles(default_delay))
            self.delay_settings[0] = clock_cycles(global_delay)

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

    def binarize_circuit(self, lru_cache=True, bypass=False):
        """Generate binary representation of all PulseData objects.
           Used primarily """
        if not self.compiled:
            self.construct_circuit(self.file, override_dict=self.override_dict)
        self.binary_data = defaultdict(list)
        circ_main = GateSlice(num_channels=self.CHANNEL_NUM)
        self.recursive_append_and_expand(self.slice_list, circ_main)
        self.apply_delays(self.delay_settings, circ_main=circ_main)
        for ch, pd_list in circ_main.channel_data.items():
            for pd in pd_list:
                self.binary_data[ch].append(pd.binarize(lru_cache=lru_cache, bypass=bypass))
        return self.binary_data

    def apply_delays(self, delay_settings=None, circ_main=None):
        """Apply delay after all triggers to account for coarse delays between channels.
           Primarily, this function serves to match AOM turn on times in counter-propagating
           configurations, where the AOMs and associated electronics might not be matched.
           However, this function is designed to support independent delays for each channel."""
        if delay_settings is None:
            return
        if circ_main is None:
            circ_main = GateSlice(num_channels=self.CHANNEL_NUM)
            self.recursive_append(self.slice_list, circ_main)
        for ch, pd_list in circ_main.channel_data.items():
            for pd in pd_list:
                if pd.waittrig:
                    pd.delay = delay_settings[ch]

    def streaming_data(self, channels=None):
        """Generate the binary data for direct streaming (bypass mode)"""
        self.binarize_circuit(lru_cache=True, bypass=True)
        if channels is None:
            channels = list(range(self.CHANNEL_NUM))
        bytelist = []
        for ch in channels:
            bytelist.extend(self.binary_data[ch])
        sorted_bytelist = timesort_bytelist(flatten(bytelist))
        return sorted_bytelist

    def walk_slice_basic(self, pd):
        for k,v in pd.channel_data.items():
            new_key = hash(tuple(v))
            self.unique_gates[k][new_key] = v
            self.unique_gate_identifiers[k][new_key] += 1
            self.gate_sequence_hashes[k].append(new_key)

    def walk_slice(self, pd, hl, reps=1):
        """Recursively walk through a slice list, including Loop objects
         to map out the PulseData hashes that serve as gate sequence ids"""
        hash_list = hl
        if isinstance(pd, GateSlice):
            for k,v in pd.channel_data.items():
                new_key = hash(tuple(v))
                self.unique_gates[k][new_key] = v
                self.unique_gate_identifiers[k][new_key] += reps
                hash_list[k].append(new_key)
        elif isinstance(pd, Loop):
            hash_list_inner = defaultdict(list)
            for p in pd:
                self.walk_slice(p, hl=hash_list_inner, reps=pd.repeats*reps)
            for k, v in hash_list_inner.items():  # only repeat the gate ids after walking a Loop
                hash_list[k].extend(v*pd.repeats)
        else:
            for p in pd:
                self.walk_slice(p, hl=hash_list, reps=reps)

    def extract_gates(self):
        """Define Gates for LUT packing based on each channel
           in each GateSlice appearing in self.slice_list"""
        self.unique_gates = defaultdict(dict)
        self.unique_gate_identifiers = defaultdict(lambda: defaultdict(int))
        self.gate_sequence_hashes = defaultdict(list)
        self.gate_sequence_ids = defaultdict(list)
        self.walk_slice(self.slice_list, hl=self.gate_sequence_hashes)
        self.ordered_gate_identifiers = dict()
        for chid in range(self.CHANNEL_NUM):
            self.ordered_gate_identifiers[chid] = dict()
            for i, (k,_) in enumerate(sorted(self.unique_gate_identifiers[chid].items(), key=lambda x: x[1], reverse=True)):
                self.ordered_gate_identifiers[chid][i] = k
            inverted_ordered_gids = {v:k for k,v in self.ordered_gate_identifiers[chid].items()}
            self.gate_sequence_ids[chid] = [inverted_ordered_gids[el] for el in self.gate_sequence_hashes[chid]]

    def generate_lookup_tables(self):
        """Construct the LUT data in an intermediate representation. The outputs
           are in a human readable format with the exception of the raw pulse data"""
        self.PLUT_data = defaultdict(list)
        self.MMAP_data = defaultdict(dict)
        self.GLUT_data = defaultdict(dict)
        for ch in range(self.CHANNEL_NUM):
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

    def generate_programming_data(self):
        """Convert the LUT programming IR representations to bytecode"""
        self.PLUT_bin = defaultdict(list)
        self.MMAP_bin = defaultdict(list)
        self.GLUT_bin = defaultdict(list)
        self.GSEQ_bin = defaultdict(list)
        for ch in range(self.CHANNEL_NUM):
            self.PLUT_bin[ch] = programPLUT({v:i for i,v in enumerate(self.PLUT_data[ch])}, ch)
            self.MMAP_bin[ch] = programSLUT(self.MMAP_data[ch], ch)
            self.GLUT_bin[ch] = programGLUT(self.GLUT_data[ch], ch)
            self.GSEQ_bin[ch] = gateSequenceBytes(self.gate_sequence_ids[ch], ch)

    def compile(self):
        """Compile the circuit, starting from parsing the jaqal file"""
        if self.file is None and self.code_literal is None:
            raise CircuitCompilerException("Need an input file!")
        self.construct_circuit(self.file, override_dict=self.override_dict)
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
        for bbind in range(0, self.CHANNEL_NUM, 8):
            prog_data = []
            seq_data = []
            packet_data = []
            for chnm in range(bbind, min(bbind+8, self.CHANNEL_NUM)):
                if channel_mask is None or (1 << chnm) & channel_mask:
                    for pd in self.last_packet_pulse_data(chnm):
                        packet_data.extend(pd.binarize(bypass=True, lru_cache=False))
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
           all channels up to self.CHANNEL_NUM"""
        if not self.compiled:
            self.compile()
        if channel_mask is None:
            channel_mask = 2**self.CHANNEL_NUM-1
        self.final_byte_dict = defaultdict(list)
        self.programming_data = list()
        self.sequence_data = list()
        if self.CHANNEL_NUM > 8:
            for bbind in range(0,self.CHANNEL_NUM,8):
                board_programming_data = list()
                board_sequence_data = list()
                for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
                                           for ch in range(bbind, min(bbind+8, self.CHANNEL_NUM))
                                           if (1 << ch) & channel_mask):
                    board_programming_data.extend(bindata[0])
                for (bindata) in zip_longest(*list(self.GSEQ_bin[ch] for ch in range(bbind, min(bbind+8, self.CHANNEL_NUM))
                                             if (1 << ch) & channel_mask)):
                    if bindata:
                        board_sequence_data.extend(bindata)
                self.programming_data.append(board_programming_data)
                self.sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
                                       for ch in range(self.CHANNEL_NUM)
                                       if (1 << ch) & channel_mask):
                board_programming_data.extend(bindata[0])
            for (bindata) in zip_longest(*list(self.GSEQ_bin[ch] for ch in range(self.CHANNEL_NUM)
                                               if (1 << ch) & channel_mask)):
                if bindata:
                    board_sequence_data.extend(bindata)
            self.programming_data.append(board_programming_data)
            self.sequence_data.append(board_sequence_data)
        return self.programming_data, self.sequence_data

    def get_prepare_all_indices(self):
        self.prepare_all_hashes = dict()
        self.prepare_all_gids = dict()
        gslice = GateSlice(num_channels=self.CHANNEL_NUM)
        if not hasattr(self.pulse_definition, 'gate_'+self.initialize_gate_name):
            raise CircuitCompilerException(f"Pulse definition has no gate named gate_{self.initialize_gate_name}")
        gate_data = getattr(self.pulse_definition, 'gate_'+self.initialize_gate_name)(self.CHANNEL_NUM)
        if gate_data is not None:
            for pd in gate_data:
                if pd.dur > 3:
                    gslice.channel_data[pd.channel].append(pd)
        for ch, gsdata in gslice.channel_data.items():
            prep_hash = hash(tuple(gsdata))
            self.prepare_all_hashes[ch] = prep_hash
            inverted_gid_hashes = {v: k for k, v in self.ordered_gate_identifiers[ch].items()}
            gid = inverted_gid_hashes.get(prep_hash, None)
            if gid is None:
                raise CircuitCompilerException(f"Unable to find hash for {self.initialize_gate_name}")
            self.prepare_all_gids[ch] = gid
        return gslice

    def generate_gate_sequence_from_index(self, ind):
        self.get_prepare_all_indices()
        partial_gs_ids = dict()
        partial_GSEQ_bin = dict()
        for ch, gidlist in self.gate_sequence_ids.items():
            partial_gs_ids[ch] = get_tail_from_index(self.prepare_all_gids[ch], gidlist, ind)
            partial_GSEQ_bin[ch] = gateSequenceBytes(partial_gs_ids[ch], ch)
        return partial_GSEQ_bin

    def partial_sequence_bytecode(self, channel_mask=None, starting_index=0):
        """Reduced form of bytecode function that only generates the sequence data from a given index"""
        if not self.compiled:
            self.compile()
        if channel_mask is None:
            channel_mask = 2**self.CHANNEL_NUM-1
        self.final_byte_dict = defaultdict(list)
        programming_data = list()
        sequence_data = list()
        partial_GSEQ_bin = self.generate_gate_sequence_from_index(starting_index)
        if self.CHANNEL_NUM > 8:
            for bbind in range(0,self.CHANNEL_NUM,8):
                board_programming_data = list()
                board_sequence_data = list()
                #for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
                                           #for ch in range(bbind, min(bbind+8, self.CHANNEL_NUM))
                                           #if (1 << ch) & channel_mask):
                    #board_programming_data.extend(bindata[0])
                for (bindata) in zip_longest(*list(partial_GSEQ_bin[ch] for ch in range(bbind, min(bbind+8, self.CHANNEL_NUM))
                                                   if (1 << ch) & channel_mask)):
                    if bindata:
                        board_sequence_data.extend(bindata)
                programming_data.append(board_programming_data)
                sequence_data.append(board_sequence_data)
        else:
            board_programming_data = list()
            board_sequence_data = list()
            #for bindata in zip_longest(self.GLUT_bin[ch]+self.MMAP_bin[ch]+self.PLUT_bin[ch]
                                       #for ch in range(self.CHANNEL_NUM)
                                       #if (1 << ch) & channel_mask):
                #board_programming_data.extend(bindata[0])
            for (bindata) in zip_longest(*list(partial_GSEQ_bin[ch] for ch in range(self.CHANNEL_NUM)
                                               if (1 << ch) & channel_mask)):
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
    inlist = list(filter(lambda x: len(x) > 0, instr.partition('//')[0].split(' ')))
    if inlist and inlist[0].strip() == 'let':
        return (inlist[1].strip(), inlist[2].strip())
    return None

def split_usepulses(instr):
    """Temporary usepulses parser for specifying pulse definition class.
       This is a placeholder until imports are resolved, and specifically
       looks for a commented line to avoid problems with parsing."""
    pd = instr.partition('//')[2].partition('usepulses')[2].strip()
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
    output = list(filter(lambda x: x is not None, map(split_lets, jp.read_text().splitlines())))
    if return_defaults:
        return {name: float_or_int(val) for (name, val) in output}
    return list(list(zip(*output))[0])

def get_gate_pulse_name(filename):
    """Get the gate pulse class name specified by a
      '//usepulses MyGatePulseClass' call in the jaqal file"""
    jp = Path(filename)
    output = list(filter(lambda x: x is not None, map(split_usepulses, jp.read_text().splitlines())))
    if output:
        return output[0]
    return None

def get_tail_from_index(elem, gidlist, ind):
    from itertools import compress, count
    gid_ind = list(compress(count(), map(lambda x: x == elem, gidlist)))[ind]
    return gidlist[gid_ind:]



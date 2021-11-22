from enum import Enum

ENDIANNESS = "little"
MAXLEN = 256 // 8  # Number of bytes in a single transfer

# LUT Address Widths
# These values contain the number of bits used for an address for a
# particular LUT. The address width is the same for reading and writing
# data for the Pulse LUT (PLUT) and the Sequence LUT (SLUT), but the
# address width is asymmetric for the Gate LUT (GLUT). This is because
# the read address is partly completed by external hardware inputs and
# the read address size (GLUTW) is thus smaller than the write address
# size (GPRGW) used to program the GLUT
GPRGW = 12  # Gate LUT write address width
GLUTW = 9  # Gate LUT read address width
SLUTW = 12  # Sequence LUT address width
PLUTW = 12  # Pulse LUT address width

# Ancilla readout extends the address space for readout from the GLUT.
# While the hardware supports a more free form approach to how one might
# choose to program the GLUT for ancilla readout purposes, combining
# normal gate sequences with ancilla readout sequences require distinguishing
# between the two cases. This is currently achieved by enforcing that the
# MSB of the physical GLUT address be set to one to mark the readout as
# pertaining to an ancilla. In this case an ancilla readout of all zeros
# will produce an output that is distinguished from the normal streaming case.
# The ANCILLA_STATE_LSB also sets the offset for ancilla readout bits
# which form the upper part of the GLUT address. This may change based on
# the number of supported hardware input lines for ancilla readout but sets
# the offset for a GLUT programming word for where the LSB of the hardware
# input register should start.
ANCILLA_COMPILER_TAG_BIT = 11  # bit used for tagging an ancilla sequence in GLUT
ANCILLA_STATE_LSB = 7  # number of bits to shift readout result in gate id

# Metadata values for the programming mode to indicate target LUT
PROGPLUT = 3
PROGSLUT = 2
PROGGLUT = 1

# Metadata LSB locations associated used to communicate the type of information
# carried in the rest of the data word. Programming/streaming modes are defined
# by 3 bits starting at PROG_MODE_LSB, they have the following states:
#
#   000 = (Not Used)
#   001 = Program GLUT
#   010 = Program SLUT
#   011 = Program PLUT
#   100 = Sequence gate IDs
#   101 = Wait for Ancilla trigger, and sequence gate IDs based on readout
#   110 = Continue sequencing based on previous ancilla data
#   111 = Bypass streaming mode
#
GSEQ_ENABLE_LSB = 247  # LSB for indicating gate readout mode
ANCILLA_CONTINUE_LSB = 246  # Continue gates from last ancilla measurement
ANCILLA_WAIT_LSB = 245  # Wait for ancilla measurement
PROG_MODE_LSB = 245  # Programming mode
GLUT_BYTECNT_LSB = 236  # Number of packed GLUT programming words
SLUT_BYTECNT_LSB = 236  # Number of packed SLUT programming words
GSEQ_BYTECNT_LSB = 239  # Number of packed gate sequence identifiers
PLUT_ADDR_LSB = 229  # LSB for address word when programming PLUT
DMA_MUX_LSB = 220  # LSB for routing to channel out of DMA


# The compiler needs to incorporate a tri-state comparison for differentiating
# between different gate sequence modes, such as a conventional gate sequence
# versus a new ancilla measurement, or a continuation of an ancilla measurement
# This is captured using the following Enum type
class GateSequenceMode(Enum):
    standard = 0
    start_branch = 1
    continue_branch = 2


# The compiler needs to handle multiple boards, each board has 8 output
# channels. When the input code targets multiple boards, the channel
# number can exceed 8, in which case a separate board is indicated, but
# the channel on that board needs to be calculated modulo 8 and is
# performed by using PER_BOARD_CH_MASK
PER_BOARD_CH_MASK = 0b111  # Bit mask for channel routing for DMA MUX

# Smallest LSB used for setting packing limits on programming data.
# This depends on the above metadata LSBs
PACKING_LIMIT_LSB = min(
    GSEQ_ENABLE_LSB,
    ANCILLA_CONTINUE_LSB,
    ANCILLA_WAIT_LSB,
    PROG_MODE_LSB,
    GLUT_BYTECNT_LSB,
    SLUT_BYTECNT_LSB,
    GSEQ_BYTECNT_LSB,
    PLUT_ADDR_LSB,
    DMA_MUX_LSB,
)

# The following LSBs indicate locations of metadata used for raw data input.
# Raw data input is typically stored in the PLUT, but can also be used in
# bypass mode.
MODTYPE_LSB = 253  # Modulation type (freq/phase/amp/framerot)
SPLSHIFT_LSB = 248  # Fixed point shift for spline coefficients
AMP_FB_EN_LSB = 228  # Amplitude feedback enable (placeholder)
FRQ_FB_EN_LSB = 227  # Frequency feedback enable
OUTPUT_EN_LSB = 226  # Toggle output enable
SYNC_FLAG_LSB = 225  # Apply global synchronization
WAIT_TRIG_LSB = 224  # Wait for external trigger

# The following LSBs are specific to frame rotation metadata
APPLY_EOF_LSB = 228  # Apply frame rotation at end of pulse
CLR_FRAME_LSB = 227  # Clear frame accumulator
FWD_FRM_T1_LSB = 226  # Forward frame to tone 1
FWD_FRM_T0_LSB = 225  # Forward frame to tone 0
INV_FRM_T1_LSB = 219  # Invert sign on frame for tone 1
INV_FRM_T0_LSB = 218  # Invert sign on frame for tone 0

# Raw pulse data information is currently broken up into metadata, duration
# and 4 spline coefficients listed with bit widths from MSB to LSB as
#
# | Metadata (56) | duration (40) | U3 (40) | U2 (40) | U1 (40) | U0 (40) |
#
# Metadata in the upper 56 bits is treated by a separate routine and appended
# to the end of a final transfer
METADATA_START_LSB = 200  # Start of metadata

# Define LSBs local to upper 56 metadata bits
MODTYPE_LSB_LOC = MODTYPE_LSB - METADATA_START_LSB
SPLSHIFT_LSB_LOC = SPLSHIFT_LSB - METADATA_START_LSB
AMP_FB_EN_LSB_LOC = AMP_FB_EN_LSB - METADATA_START_LSB
APPLY_EOF_LSB_LOC = APPLY_EOF_LSB - METADATA_START_LSB
FRQ_FB_EN_LSB_LOC = FRQ_FB_EN_LSB - METADATA_START_LSB
CLR_FRAME_LSB_LOC = CLR_FRAME_LSB - METADATA_START_LSB
OUTPUT_EN_LSB_LOC = OUTPUT_EN_LSB - METADATA_START_LSB
SYNC_FLAG_LSB_LOC = SYNC_FLAG_LSB - METADATA_START_LSB
WAIT_TRIG_LSB_LOC = WAIT_TRIG_LSB - METADATA_START_LSB
DMA_MUX_OFFSET_LOC = DMA_MUX_LSB - METADATA_START_LSB
FWD_FRM_T1_LSB_LOC = FWD_FRM_T1_LSB - METADATA_START_LSB
FWD_FRM_T0_LSB_LOC = FWD_FRM_T0_LSB - METADATA_START_LSB
INV_FRM_T1_LSB_LOC = INV_FRM_T1_LSB - METADATA_START_LSB
INV_FRM_T0_LSB_LOC = INV_FRM_T0_LSB - METADATA_START_LSB

# Number of programming or gate sequence words that can be packed into a single
# transfer. PLUT programming data is always one word per transfer.
SLUT_BYTECNT = PACKING_LIMIT_LSB // (SLUTW + PLUTW)
GLUT_BYTECNT = PACKING_LIMIT_LSB // (GPRGW + 2 * SLUTW)
GSEQ_BYTECNT = PACKING_LIMIT_LSB // GLUTW

# Integer encoded representation of modulation type
FRQMOD0INT = 0
AMPMOD0INT = 1
PHSMOD0INT = 2
FRQMOD1INT = 3
AMPMOD1INT = 4
PHSMOD1INT = 5
FRMROT0INT = 6
FRMROT1INT = 7

# One-hot encoded representation of modulation type
FRQMOD0 = 1 << FRQMOD0INT
AMPMOD0 = 1 << AMPMOD0INT
PHSMOD0 = 1 << PHSMOD0INT
FRQMOD1 = 1 << FRQMOD1INT
AMPMOD1 = 1 << AMPMOD1INT
PHSMOD1 = 1 << PHSMOD1INT
FRMROT0 = 1 << FRMROT0INT
FRMROT1 = 1 << FRMROT1INT

# Set the minimum clock cycles for a pulse to help avoid underflows. This time
# is determined by state machine transitions for loading another gate, but does
# not account for serialization of pulse words.
MINIMUM_PULSE_CLOCK_CYCLES = 4

import time
import asyncio

from jaqalpaw.bytecode.encoding_parameters import (
    CLR_FRAME_LSB,
    APPLY_EOF_LSB,
    ANCILLA_COMPILER_TAG_BIT,
    ANCILLA_STATE_LSB,
    FWD_FRM_T0_LSB,
    INV_FRM_T0_LSB,
    FRMROT0INT,
    FRMROT1INT,
)
from .byte_decoding import *


def construct_fifos(
    num_spline_fifos=8,
    num_channels=8,
    spline_fifo_depth=4,
    gate_seq_fifo_depth=32,
    dma_depth=256,
):
    spline_fifos = [
        [asyncio.Queue(maxsize=spline_fifo_depth) for _ in range(num_spline_fifos)]
        for _ in range(num_channels)
    ]
    gseq_fifos = [
        asyncio.Queue(maxsize=gate_seq_fifo_depth) for _ in range(num_channels)
    ]
    dma_queue = asyncio.Queue(maxsize=dma_depth)
    return spline_fifos, gseq_fifos, dma_queue


async def DMA_arbiter(name, queue, data_output_queues):
    """Send data to the correct channel (or gate sequencer input FIFO) based on metadata"""
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder="little", signed=False)
        channel = (data >> DMA_MUX_LSB) & 0b111
        await data_output_queues[channel].put(raw_data)
        queue.task_done()


async def gate_seq_arbiter(name, queue, data_output_queues):
    """Performs the same functions as the hardware GateSequencer IP cores.
    Input words are 256 bits, and are parsed and treated accordingly depending on the
    metadata tags in the raw data in order to program LUTs or run gate sequences etc...
    If the LUTs are not being programmed, the resulting output is sent to the spline engine FIFOs"""
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder="little", signed=False)
        mod_type = (data >> MODTYPE_LSB) & 0b111
        prog_mode = (data >> PROG_MODE_LSB) & 0b111
        if prog_mode == 0b111:
            await data_output_queues[mod_type].put(raw_data)
        elif prog_mode == 0b001:
            parse_GLUT_prog_data(data)
        elif prog_mode == 0b010:
            parse_SLUT_prog_data(data)
        elif prog_mode == 0b011:
            parse_PLUT_prog_data(raw_data)
        elif prog_mode == 0b100 or prog_mode == 0b101 or prog_mode == 0b110:
            if prog_mode == 0b101 or prog_mode == 0b110:
                # at the very least we'll need the streamed data to be augmented
                # by the tag bit. Additional cases can be applied by ORing the
                # "OR address", oraddr, (representing external hardware input)
                # via some (binary) state by adding the following line after
                # oraddr is initially set to 1 << ANCILLA_COMPILER_TAG_BIT
                #
                #     oraddr |= state << ANCILLA_STATE_LSB
                #
                # to execute a different branch. But this is not yet worked into
                # the emulator in a way that supports a sequence of ancilla
                # measurement states.
                oraddr = 1 << ANCILLA_COMPILER_TAG_BIT
            else:
                oraddr = 0
            for gs_data in parse_gate_seq_data(data, oraddr=oraddr):
                new_mod_type = (
                    int.from_bytes(gs_data, byteorder="little", signed=False)
                    >> MODTYPE_LSB
                ) & 0b111
                await data_output_queues[new_mod_type].put(gs_data)
        queue.task_done()


async def spline_engine(
    name,
    queue,
    time_list,
    data_list,
    waittrig_list,
    enablemask_list,
    fwd_frame0_mask_list,
    inv_frame0_mask_list,
    fwd_frame1_mask_list,
    inv_frame1_mask_list,
):
    """Converts the spline coefficients to a format that can be passed into a SplineEngine emulator,
    which generates the corresponding output and stores the data in time_list and data_list for
    plotting and/or inspecting the data"""
    eof_data = 0
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder="little", signed=False)
        waittrig = (data >> WAIT_TRIG_LSB) & 0b1
        enablemask = (data >> OUTPUT_EN_LSB) & 0b1
        fwd_frame0_mask = 0
        inv_frame0_mask = 0
        fwd_frame1_mask = 0
        inv_frame1_mask = 0
        mod_type = (data >> MODTYPE_LSB) & 0b111
        if mod_type == FRMROT0INT:
            fwd_frame0_mask = (data >> FWD_FRM_T0_LSB) & 0b11
            inv_frame0_mask = (data >> INV_FRM_T0_LSB) & 0b11
        elif mod_type == FRMROT1INT:
            fwd_frame1_mask = (data >> FWD_FRM_T0_LSB) & 0b11
            inv_frame1_mask = (data >> INV_FRM_T0_LSB) & 0b11
        shift = (data >> SPLSHIFT_LSB) & 0b11111
        channel = (data >> DMA_MUX_LSB) & 0b111
        reset_accum = (data >> CLR_FRAME_LSB) & 0b1
        apply_at_eof = (data >> APPLY_EOF_LSB) & 0b1
        dur, U0, U1, U2, U3 = parse_bypass_data(raw_data)
        # Convert binary values to real-unit equivalents for monitoring
        # dur += TIMECORR+0#+4
        dur_real = convert_time_from_clock_cycles(dur)
        U0_real = mod_type_dict[mod_type]["realConvFunc"](U0)
        U1_real = mod_type_dict[mod_type]["realConvFunc"](U1)
        U2_real = mod_type_dict[mod_type]["realConvFunc"](U2)
        U3_real = mod_type_dict[mod_type]["realConvFunc"](U3)
        # This if statement is not absolutely necessary but reduces number of points to plot, forcing
        # the function to always jump to the else clause will produce the same output and is more
        # consistent with how the hardware is actually operating.
        if U1 == 0 and U2 == 0 and U3 == 0:
            time_list.append(time_list[-1] + dur)
            if mod_type in (
                FRMROT0INT,
                FRMROT1INT,
            ):  # then we have a z rotation which must accumulate from old values
                if reset_accum:
                    last_val = 0
                    eof_data = 0
                else:
                    last_val = data_list[-1]
                if apply_at_eof:
                    data_list.append(last_val + eof_data)
                    eof_data = U0_real
                else:
                    data_list.append(last_val + U0_real + eof_data)
                    eof_data = 0
                if mod_type == FRMROT0INT:
                    fwd_frame0_mask_list.append(fwd_frame0_mask)
                    inv_frame0_mask_list.append(inv_frame0_mask)
                else:
                    fwd_frame1_mask_list.append(fwd_frame1_mask)
                    inv_frame1_mask_list.append(inv_frame1_mask)
            else:
                data_list.append(U0_real)
                waittrig_list[-1] = waittrig
                waittrig_list.append(0)
                enablemask_list[-1] = enablemask
                enablemask_list.append(0)
            print(
                f"channel: {channel}, type: {mod_type}, Duration: {dur_real} s, U0: {U0_real}, U1: {U1_real}, U2: {U2_real}, U3: {U3_real}"
            )
        else:
            # Bit shifting is done to enhance precision within firmware
            U1_shift = U1
            U2_shift = U2
            U3_shift = U3
            # Calculate the same for real values for monitoring purposes only
            U1_rshift = U1_real / (1 << shift)
            U2_rshift = U2_real / (1 << (shift * 2))
            U3_rshift = U3_real / (1 << (shift * 3))
            # Pack the coefficients in a format that can be handled by the spline engine emulator
            coeffs = np.zeros((4, 1))
            coeffs[0, 0] = U3_shift
            coeffs[1, 0] = U2_shift
            coeffs[2, 0] = U1_shift
            coeffs[3, 0] = U0
            # The additional 3 clock cycles are related to a subtle hardware issue
            xdata = np.array(list(range(dur))) + 1
            spline_data = pdq_spline(coeffs, [0], nsteps=dur, shift=shift)
            spline_data_real = list(
                map(mod_type_dict[mod_type]["realConvFunc"], spline_data)
            )
            xdata_real = list(map(lambda x: time_list[-1] + x, xdata))
            time_list.extend(xdata_real[:])
            last_val = data_list[-1]
            if mod_type in (
                FRMROT0INT,
                FRMROT1INT,
            ):  # then we have a z rotation which must accumulate from old values
                if reset_accum:
                    last_val = 0
                    eof_data = 0
                data_list.extend(last_val + eof_data + np.array(spline_data_real))
                eof_data = 0
                if mod_type == FRMROT0INT:
                    fwd_frame0_mask_list.extend([fwd_frame0_mask] * len(xdata_real))
                    inv_frame0_mask_list.extend([inv_frame0_mask] * len(xdata_real))
                else:
                    fwd_frame1_mask_list.extend([fwd_frame1_mask] * len(xdata_real))
                    inv_frame1_mask_list.extend([inv_frame1_mask] * len(xdata_real))
            else:
                data_list.extend(spline_data_real)
            print(
                f"channel: {channel}, type: {mod_type}, Duration: {dur_real} s, "
                f"U0: {U0_real}, U1: {U1_rshift}, U2: {U2_rshift}, U3: {U3_rshift}"
            )
        # For good measure, wait for the duration encoded in the raw data, for more accurate emulation, this duration
        # should be scaled so as to reduce the influence imposed by computational delay
        # await asyncio.sleep(dur_real)
        queue.task_done()

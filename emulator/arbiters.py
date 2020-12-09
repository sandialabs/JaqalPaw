import time
import asyncio

from bytecode.encoding_parameters import CLR_FRAME_LSB
from .uram import GADDRW, SADDRW, PADDRW, GLUT, SLUT, PLUT, URAM
from .byte_decoding import *
import copy


def construct_fifos(num_spline_fifos=8, num_channels=8, spline_fifo_depth=4, gate_seq_fifo_depth=32, dma_depth=256):
    spline_fifos = [[asyncio.Queue(maxsize=spline_fifo_depth) for _ in range(num_spline_fifos)] for _ in range(num_channels)]
    gseq_fifos = [asyncio.Queue(maxsize=gate_seq_fifo_depth) for _ in range(num_channels)]
    dma_queue = asyncio.Queue(maxsize=dma_depth)
    return spline_fifos, gseq_fifos, dma_queue


async def DMA_arbiter(name, queue, data_output_queues):
    """Send data to the correct channel (or gate sequencer input FIFO) based on metadata"""
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder='little', signed=False)
        channel = (data >> (DMA_MUX_OFFSET)) & 0b111
        await data_output_queues[channel].put(raw_data)
        queue.task_done()


async def gate_seq_arbiter(name, queue, data_output_queues):
    """Performs the same functions as the hardware GateSequencer IP cores.
       Input words are 256 bits, and are parsed and treated accordingly depending on the
       metadata tags in the raw data in order to program LUTs or run gate sequences etc...
       If the LUTs are not being programmed, the resulting output is sent to the spline engine FIFOs"""
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder='little', signed=False)
        mod_type = (data >> (MODTYPE_LSB)) & 0b111
        prog_mode = (data >> (PROG_MODE_OFFSET)) & 0b111
        if prog_mode == 0b111:
            await data_output_queues[mod_type].put(raw_data)
        elif prog_mode == 0b001:
            parse_GLUT_prog_data(data)
        elif prog_mode == 0b010:
            parse_SLUT_prog_data(data)
        elif prog_mode == 0b011:
            parse_PLUT_prog_data(raw_data)
        elif prog_mode == 0b100 or prog_mode == 0b101 or prog_mode == 0b110:
            for gs_data in parse_gate_seq_data(data):
                new_mod_type = (int.from_bytes(gs_data, byteorder='little', signed=False) >> (MODTYPE_LSB)) & 0b111
                await data_output_queues[new_mod_type].put(gs_data)
        queue.task_done()


async def spline_engine(name, queue, time_list, data_list, waittrig_list, enablemask_list):
    """Converts the spline coefficients to a format that can be passed into a SplineEngine emulator,
       which generates the corresponding output and stores the data in time_list and data_list for
       plotting and/or inspecting the data"""
    while True:
        raw_data = await queue.get()
        data = int.from_bytes(raw_data, byteorder='little', signed=False)
        waittrig = (data >> WAIT_TRIG_LSB) & 0b1
        enablemask = (data >> OUTPUT_EN_LSB) & 0b11
        mod_type = (data >> MODTYPE_LSB) & 0b111
        shift = (data >> SPLSHIFT_LSB) & 0b11111
        channel = (data >> DMA_MUX_OFFSET) & 0b111
        reset_accum = (data >> CLR_FRAME_LSB) & 0b1
        dur, U0, U1, U2, U3 = parse_bypass_data(raw_data)
        # Convert binary values to real-unit equivalents for monitoring
        #dur += TIMECORR+0#+4
        dur_real = convert_time_from_clock_cycles(dur)
        U0_real = mod_type_dict[mod_type]['realConvFunc'](U0)
        U1_real = mod_type_dict[mod_type]['realConvFunc'](U1)
        U2_real = mod_type_dict[mod_type]['realConvFunc'](U2)
        U3_real = mod_type_dict[mod_type]['realConvFunc'](U3)
        # This if statement is not absolutely necessary but reduces number of points to plot, forcing
        # the function to always jump to the else clause will produce the same output and is more
        # consistent with how the hardware is actually operating.
        if U1 == 0 and U2 == 0 and U3 == 0:
            time_list.append(time_list[-1]+dur)
            if mod_type > 5: # then we have a z rotation which must accumulate from old values
                last_val = data_list[-1] if not reset_accum else 0
                data_list.append(last_val+U0_real)
            else:
                data_list.append(U0_real)
                waittrig_list[-1] = waittrig
                waittrig_list.append(0)
                enablemask_list[-1] = enablemask
                enablemask_list.append(0)
            print(f"channel: {channel}, type: {mod_type}, Duration: {dur_real} s, U0: {U0_real}, U1: {U1_real}, U2: {U2_real}, U3: {U3_real}")
        else:
            # Bit shifting is done to enhance precision within firmware
            U1_shift = U1/(2**(shift*1))
            U2_shift = U2/(2**(shift*2))
            U3_shift = U3/(2**(shift*3))
            # Calculate the same for real values for monitoring purposes only
            U1_rshift = U1_real/(2**shift)
            U2_rshift = U2_real/(2**(shift*2))
            U3_rshift = U3_real/(2**(shift*3))
            # Pack the coefficients in a format that can be handled by the spline engine emulator
            coeffs = np.zeros((4,1))
            coeffs[0,0] = U3_shift
            coeffs[1,0] = U2_shift
            coeffs[2,0] = U1_shift
            coeffs[3,0] = U0
            # The additional 3 clock cycles are related to a subtle hardware issue
            xdata = np.array(list(range(dur)))+1
            spline_data = pdq_spline(coeffs, [0], nsteps=dur)
            spline_data_real = list(map(mod_type_dict[mod_type]['realConvFunc'], spline_data))
            xdata_real = list(map(lambda x: time_list[-1]+x, xdata))
            time_list.extend(xdata_real[1:])
            last_val = data_list[-1]
            if mod_type > 5: # then we have a z rotation which must accumulate from old values
                if reset_accum:
                    last_val = 0
                data_list.extend(last_val+np.array(spline_data_real))
            else:
                data_list.extend(spline_data_real)
            print(f"channel: {channel}, type: {mod_type}, Duration: {dur_real} s, "
                  f"U0: {U0_real}, U1: {U1_rshift}, U2: {U2_rshift}, U3: {U3_rshift}")
        # For good measure, wait for the duration encoded in the raw data, for more accurate emulation, this duration
        # should be scaled so as to reduce the influence imposed by computational delay
        #await asyncio.sleep(dur_real)
        queue.task_done()


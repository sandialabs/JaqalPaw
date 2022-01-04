from jaqalpaq.core import CircuitBuilder

from .uram import *
from .arbiters import *
import numpy as np
import copy

# from emulator.byte_decoding import decode_word, GLUT, SLUT, PLUT, mod_type_dict#, tree
# master_data_record = {c: {d:{'time':copy.copy([0]), 'data':copy.copy([0])} for d in range(8)} for c in range(8)}


async def firmware_emulator(
    data_list, num_channels=8, num_dtypes=8, master_data_record=None
):
    r"""
    The firmware_emulator is the main function that encodes the pipeline structure in the hardware.
    Any hardware feature that stores data (such as DMA or FIFOs) are described by asyncio.Queue objects.
    DMA transfers put the data into the main pipeline, called DMA Queue and the data is subsequently consumed
    by lower level elements. Data passing through the DMA Queue is passed to 8 FIFOs corresponding to different
    output channels as determined from the metadata encoded in a data word. This data is then passed into individual
    gate sequencers, which decide to program internal LUTs, read values out of those LUTs, or bypass the LUTs and
    feed the data directly into the final stage of FIFOs depending on the metadata. This final stage of FIFOs is broken
    up into different parameters, which are frequency, phase, amplitude, and z-rotations (sometimes referred to as frame
    rotations in the source code), with duplicate parameters for both tones, giving a total of 8 FIFOs per channel.
    These FIFOs are directly fed into spline engines, which ultimately parse the data words and provide relevant output
    to the DDS cores. The approximate structure of the system is shown below


                                            DMA Queue
                                                |
      channel fifo     0______1______2______3___|___4______5______6______7
                       |      |      |      |       |      |      |      |
                     GSeq0  GSeq1  GSeq2  GSeq3   GSeq4  GSeq5  GSeq6  GSeq7
                      /|\    /|\    /|\    /|\     /|\    /|\    /|\    /|\
                     / | \  / | \  / | \  / | \   / | \  / | \  / | \  / | \
       spline fifo   0...7  0...7  0...7  0...7   0...7  0...7  0...7  0...7

    Because these FIFOs need to be represented by asyncio.Queue objects, they need to be created by a top-level
    function and thus all Queues are defined within this function.

    In the hardware, spline engines will throttle the data based on the amount of time encoded into each data word. This
    behavior is approximated by an asyncio.sleep() call towards the end of the spline_engine() function, but should
    probably scale this time to get a more accurate measure of delays imposed by the spline engines due to the natural
    computation time needed to generate realistic output. In the hardware, the effective computational overhead leads
    to a natural latency that is constant across all spline engines and can be neglected.

    This function sets up the data queues for all the hardware elements, takes raw byte strings that are equivalent to
    the binary data sent to the RFSoC, and waits for all of the data to be consumed. Special functionality beyond that
    is handled by some of the other async functions:

    DMA_arbiter     : delegates data words to the correct channel based on those words' metadata
    gate_seq_arbiter : either encodes or decodes the LUTs and delegates any output data to the correct spline engine FIFO
    spline_engine   : consumes incoming data from its dedicated FIFO and reproduces the spline output

    The spline_engine also takes additional inputs which are (nearly) empty lists in the global master_data_record
    object that is used to store the spline engine output data for inspection.

    Note: some functionality has been put in place to more accurately emulate potential error cases such as starving
          the FIFOs during what should otherwise be a contiguous stream of data. In this version of the code, a fully
          featured representation of these cases is lacking. In order to implement the correct behavior, one needs to
          scale the sleep time near the end of SplineEngine() and more importantly add a feature that functions like
          an input trigger. Data is necessarily streamed in serially, but once the FIFOs are filled, the time spent
          waiting for the next data word (as contained in each data word) should typically be longer than the time it
          takes to push more data into the FIFO. However, this cannot be done in parallel initially and needs to wait
          for an external impulse to set everything running. There are probably better methods that could be employed
          to better track these kinds of errors, but for now I leave it to the user to implement such improvements.
    """
    # Create all of the queue objects, and optionally specify their depth
    spline_fifos, gseq_fifos, dma_queue = construct_fifos(
        num_spline_fifos=num_dtypes,
        num_channels=num_channels,
        spline_fifo_depth=16,  # 256,
        gate_seq_fifo_depth=32,  # 512,
        dma_depth=2**20,
    )

    # generate fifo tasks
    tasks = []
    task = asyncio.Task(DMA_arbiter(f"DMA-arbiter-{0}", dma_queue, gseq_fifos))
    tasks.append(task)
    for nc in range(num_channels):  # 8 gate sequencers, one per channel
        task = asyncio.Task(
            gate_seq_arbiter(
                f"gate-seq-arbiter-ch{nc}",  # name for bookkeeping purposes
                gseq_fifos[nc],  #
                spline_fifos[nc],
            )
        )
        tasks.append(task)
        for nd in range(
            num_dtypes
        ):  # 64 spline engines, 8 per channel (4 parameters, 2 tones)
            task = asyncio.Task(
                spline_engine(
                    f"spline-engine-ch{nc}-d{nd}",
                    spline_fifos[nc][nd],
                    master_data_record[nc][nd]["time"],  # x axis plot data
                    master_data_record[nc][nd]["data"],  # y axis plot data
                    master_data_record[nc][nd]["waittrig"],
                    master_data_record[nc][nd]["enablemask"],
                    master_data_record[nc][nd]["fwd_frame0_mask"],
                    master_data_record[nc][nd]["inv_frame0_mask"],
                    master_data_record[nc][nd]["fwd_frame1_mask"],
                    master_data_record[nc][nd]["inv_frame1_mask"],
                )
            )
            tasks.append(task)

    # feed binary data into DMA queue
    for dataw in data_list:
        await dma_queue.put(dataw)

    # Wait until the queue is fully processed.
    started_at = time.monotonic()
    await dma_queue.join()
    for nc in range(num_channels):
        await gseq_fifos[nc].join()
        for nd in range(num_dtypes):
            await spline_fifos[nc][nd].join()
    total_slept_for = time.monotonic() - started_at

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)

    print("====")
    print(
        f"Spline Engines experienced a net (parallel) delay of {total_slept_for:.2f} seconds"
    )

    for nc in range(num_channels):  # 8 gate sequencers, one per channel
        for nd in range(
            num_dtypes
        ):  # 64 spline engines, 8 per channel (4 parameters, 2 tones)
            master_data_record[nc][nd]["data"][:-1] = master_data_record[nc][nd][
                "data"
            ][1:]
            master_data_record[nc][nd]["waittrig"][:-1] = master_data_record[nc][nd][
                "waittrig"
            ][1:]
            master_data_record[nc][nd]["enablemask"][:-1] = master_data_record[nc][nd][
                "enablemask"
            ][1:]
            master_data_record[nc][nd]["fwd_frame0_mask"][:-1] = master_data_record[nc][
                nd
            ]["fwd_frame0_mask"][1:]
            master_data_record[nc][nd]["fwd_frame1_mask"][:-1] = master_data_record[nc][
                nd
            ]["fwd_frame1_mask"][1:]
            master_data_record[nc][nd]["inv_frame0_mask"][:-1] = master_data_record[nc][
                nd
            ]["inv_frame0_mask"][1:]
            master_data_record[nc][nd]["inv_frame1_mask"][:-1] = master_data_record[nc][
                nd
            ]["inv_frame1_mask"][1:]


from collections import defaultdict

tree = lambda: defaultdict(tree)


def chunk_data_direct(data, chunksize=65536):
    chunks = len(data)
    chunk_size = 32 * chunksize
    return [data[i : i + chunk_size] for i in range(0, chunks, chunk_size)]


def trigger_events(input_bytes):
    master_data_record = {
        c: {
            d: {
                "time": copy.copy([0]),
                "data": copy.copy([0]),
                "waittrig": copy.copy([0]),
                "enablemask": copy.copy([0]),
            }
            for d in range(8)
        }
        for c in range(8)
    }
    retlist = chunk_data_direct(input_bytes, chunksize=1)

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        firmware_emulator(
            retlist, num_channels=8, master_data_record=master_data_record
        )
    )
    loop.close()
    ch_list = list()
    for chnm in range(8):
        for n in range(1):  # wait trig masks apply to all parameters
            tlist = []
            for idx, t in enumerate(master_data_record[chnm][n]["time"]):
                if master_data_record[chnm][n]["waittrig"][idx]:
                    tlist.append(t)
            ch_list.append(tlist)
    return ch_list


def print_metadata_output(input_bytes):
    master_data_record = {
        c: {
            d: {
                "time": copy.copy([0]),
                "data": copy.copy([0]),
                "waittrig": copy.copy([0]),
                "enablemask": copy.copy([0]),
            }
            for d in range(8)
        }
        for c in range(8)
    }
    retlist = chunk_data_direct(input_bytes, chunksize=1)

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        firmware_emulator(
            retlist, num_channels=8, master_data_record=master_data_record
        )
    )
    loop.close()
    for chnm in range(8):
        for n in range(1):  # wait trig masks apply to all parameters
            tlist = []
            for idx, t in enumerate(master_data_record[chnm][n]["time"]):
                if master_data_record[chnm][n]["waittrig"][idx]:
                    tlist.append(t)
            tlist_conversion = list(map(convert_time_from_clock_cycles, tlist))
            real_time = map(lambda s: "{:.3g}".format(s * 1e6), tlist_conversion)
            diff = map(lambda s: "{:.3g}".format(s * 1e6), np.diff(tlist_conversion))
            print(
                f"ch {chnm} received {len(tlist)} wait trigger events at times: {tlist} diff: {np.diff(tlist)}"
                f" real time (us): [{' '.join(list(real_time))}]"
                f" diff: [{' '.join(list(diff))}]"
            )

    for chnm in range(8):
        for tone, n in enumerate([0, 4]):  # range(1):
            elist = []
            enablemask = 0
            for idx, t in enumerate(master_data_record[chnm][n]["time"]):
                if master_data_record[chnm][n]["enablemask"][idx] != enablemask:
                    enablemask = master_data_record[chnm][n]["enablemask"][idx]
                    elist.append((t, enablemask))
            if elist:
                print(
                    f"ch {chnm} tone {tone} received {len(elist)} enable mask events at times: {elist}"
                )
                slstr = "state: "
                tstr = " time: "
                deltastr = "\u0394time: "
                lastt = 0
                for t, v in elist:
                    if v:
                        tadd = f" {t}"
                        tstr += tadd
                        slstr += "/" + "\u203e" * (len(tadd) - 1)
                        delta = f" {t-lastt}"
                        delta += " " * (len(tadd) - len(delta))
                        deltastr += delta
                    else:
                        tadd = f" {t}"
                        tstr += tadd
                        slstr += "\\" + "_" * (len(tadd) - 1)
                        delta = f" {t-lastt}"
                        delta += " " * (len(tadd) - len(delta))
                        deltastr += delta
                    lastt = t
                print(slstr)
                print(tstr)
                print(deltastr)


def plot_octet_emulator_output(
    ret, compare_lut_to_bypass=False, num_plots=8, legend=True, real_time=False
):
    from jaqalpaw.emulator.byte_decoding import (
        decode_word,
        GLUT,
        SLUT,
        PLUT,
        mod_type_dict,
    )

    import matplotlib.pyplot as plt
    import matplotlib.ticker as tic

    master_data_record = {
        c: {
            d: {
                "time": [0],
                "data": [0],
                "waittrig": [0],
                "enablemask": [0],
                "fwd_frame0_mask": [0],
                "inv_frame0_mask": [0],
                "fwd_frame1_mask": [0],
                "inv_frame1_mask": [0],
            }
            for d in range(8)
        }
        for c in range(8)
    }
    retlist = chunk_data_direct(ret, chunksize=1)

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        firmware_emulator(
            retlist, num_channels=num_plots, master_data_record=master_data_record
        )
    )
    loop.close()

    print("finished")
    print("Final LUT contents as lists, ordered by channel:")
    print("Gate LUTs:  ", GLUT)
    print("MMAP LUTs:  ", SLUT)
    print("Pulse LUTs: ", PLUT)

    mdr = master_data_record
    apply_frame_forwarding_and_inversion(mdr, num_channels=num_plots)
    print("ret len is ", len(ret))

    plot_loc_dict = {
        0: [0, 0],
        1: [0, 1],
        2: [0, 2],
        3: [1, 0],
        4: [1, 1],
        5: [1, 2],
        6: [0, 3],
        7: [1, 3],
    }

    time_scale = CLKPERIOD * 1e6 if real_time else 1
    f1, axl = plt.subplots(4, 2, sharex=True, figsize=(7.4, 4.8))
    ls_list = 2 * [
        {"linestyle": "-", "where": "post"},
        {"linestyle": "-.", "where": "post"},
        {"linestyle": "--", "where": "post"},
        {"linestyle": ":", "where": "post"},
    ]
    pltref = []
    for chnm in range(num_plots):
        for i in range(4):
            for j in range(2):
                jp, ip = plot_loc_dict[i + j * 4]
                axl[ip][jp].set_ylabel(mod_type_dict[i + j * 4]["name"])
                (pr,) = axl[ip][jp].step(
                    time_scale * np.array(mdr[chnm][i + j * 4]["time"]),
                    mdr[chnm][i + j * 4]["data"],
                    **ls_list[chnm],
                    label=f"Channel {chnm}",
                )
                if i == 0 and j == 0:
                    pltref.append(copy.copy(pr))
                plt.subplots_adjust(hspace=0.001, wspace=0.001)
                temp = tic.MaxNLocator(4)
                axl[ip][jp].yaxis.set_major_locator(temp)
                if jp == 1:
                    axl[ip][jp].yaxis.set_label_position("right")
                    axl[ip][jp].yaxis.tick_right()
                if ip == 3:
                    axl[ip][jp].set_xlabel(
                        r"Time ($\mu$s)" if real_time else "Clock Cycles"
                    )

    if legend:
        plt.subplots_adjust(bottom=0.1, right=0.7, top=0.9)
        axbox = (
            axl[0][-1]
            .get_tightbbox(f1.canvas.get_renderer())
            .transformed(axl[0][-1].transAxes.inverted())
        )
        axl[0][-1].legend(
            pltref,
            [f"Channel {cn}" for cn in range(num_plots)],
            loc="upper left",
            bbox_to_anchor=[axbox.x1, axbox.y1],
            bbox_transform=axl[0][-1].transAxes,
        )
    plt.show()


def apply_frame_forwarding_and_inversion(mdr, num_channels=8):
    """Frame data needs to be forwarded to different tones with an optional
    sign inversion. Because the frame data has a different array for update
    times to reduce overhead on plotting, as well as the fact that the data
    is generated independently in separate spline engine queues, the data needs
    to be post-processed to reflect the correct outputs. This function replaces
    the input data with the correct forwarding/inversion information based on
    the same precedence rules used in firmware. Namely, trying to forward data
    from both frames to the same tones is allowed by the hardware, but the data
    from framerot0 will take precedence."""
    for chnm in range(num_channels):
        xd = [None, None]
        yd = [None, None]
        fmask = [None, None]
        invmask = [None, None]
        for i, pind in enumerate([FRMROT0INT, FRMROT1INT]):
            xd[i] = np.array(mdr[chnm][pind]["time"])
            yd[i] = np.array(mdr[chnm][pind]["data"])
            fmask[i] = np.array(mdr[chnm][pind][f"fwd_frame{i}_mask"])
            invmask[i] = np.array(mdr[chnm][pind][f"inv_frame{i}_mask"])
        x0_fin, x1_fin, y0_fin, y1_fin = [], [], [], []
        all_xs = sorted(set(xd[0]) | set(xd[1]))
        for x in all_xs:
            fwd0m, fwd1m = 0, 0
            if x in xd[0]:
                fwd0m = fmask[0][xd[0] == x][-1]
            if x in xd[1]:
                fwd1m = fmask[1][xd[1] == x][-1]
            if 0b01 & fwd0m:
                if 0b01 & invmask[0][xd[0] == x][-1]:
                    y0_fin.append(-yd[0][xd[0] == x][-1])
                else:
                    y0_fin.append(yd[0][xd[0] == x][-1])
                x0_fin.append(x)
            elif 0b01 & fwd1m:
                if 0b01 & invmask[1][xd[1] == x][0]:
                    y0_fin.append(-yd[1][xd[1] == x][-1])
                else:
                    y0_fin.append(yd[1][xd[1] == x][-1])
                x0_fin.append(x)
            else:
                y0_fin.append(0)
                x0_fin.append(x)
            if 0b10 & fwd0m:
                if 0b10 & invmask[0][xd[0] == x][-1]:
                    y1_fin.append(-yd[0][xd[0] == x][-1])
                else:
                    y1_fin.append(yd[0][xd[0] == x][-1])
                x1_fin.append(x)
            elif 0b10 & fwd1m:
                if 0b10 & invmask[1][xd[1] == x][-1]:
                    y1_fin.append(-yd[1][xd[1] == x][-1])
                else:
                    y1_fin.append(yd[1][xd[1] == x][-1])
                x1_fin.append(x)
            else:
                y1_fin.append(0)
                x1_fin.append(x)
        mdr[chnm][FRMROT0INT]["time"] = np.array(x0_fin)
        mdr[chnm][FRMROT1INT]["time"] = np.array(x1_fin)
        mdr[chnm][FRMROT0INT]["data"] = np.array(y0_fin)
        mdr[chnm][FRMROT1INT]["data"] = np.array(y1_fin)

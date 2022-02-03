import numpy as np

# ######################################################## #
# ------------ pdq Spline Coefficient Mapping ------------ #
# ######################################################## #


def cs_mapper_int(interp_table, nsteps=409625, shift_len=16):
    """Map spline coefficients into representation that
    can be used for accumulator based reconstruction.
    Convert the data to integers for hardware and bitshift
    higher order coefficients for better resolution"""
    new_coeffs = np.zeros(interp_table.shape)
    for i in range(interp_table.shape[1]):
        tstep = 1 / nsteps if not isinstance(nsteps, list) else 1 / nsteps[i]
        new_coeffs[3, i] = float(interp_table[3, i])
        new_coeffs[2, i] = float(
            (
                interp_table[2, i] * tstep
                + interp_table[1, i] * tstep**2
                + interp_table[0, i] * tstep**3
            )
            * (1 << shift_len)
        )
        new_coeffs[1, i] = float(
            (2 * interp_table[1, i] * tstep**2 + 6 * interp_table[0, i] * tstep**3)
            * (1 << (shift_len * 2))
        )
        new_coeffs[0, i] = float(
            (6 * interp_table[0, i] * tstep**3) * (1 << (shift_len * 3))
        )
    return new_coeffs


def cs_mapper_int_auto_shift(interp_table, nsteps, apply_phase_mask=False):
    """Map spline coefficients into representation that
    can be used for accumulator based reconstruction.
    This variation automatically determines the optimal
    bit shift for a given set of spline coefficients"""
    new_coeffs = np.zeros(interp_table.shape)
    shift_len_list = []
    for i in range(interp_table.shape[1]):
        tstep = 1 / nsteps[i]

        # initial pdq mapping, ordering is the same as the default output
        # of scipy's cubic spline coefficients, where index is inversely
        # related to the order of each coefficient.
        if apply_phase_mask:
            new_coeffs[3, i] = (
                (int(interp_table[3, i]) & 0xFFFFFFFFFF) ^ 0x8000000000
            ) - 0x8000000000
        else:
            new_coeffs[3, i] = int(interp_table[3, i])
        new_coeffs[2, i] = (
            interp_table[2, i] * tstep
            + interp_table[1, i] * tstep**2
            + interp_table[0, i] * tstep**3
        )
        new_coeffs[1, i] = (
            2 * interp_table[1, i] * tstep**2 + 6 * interp_table[0, i] * tstep**3
        )
        new_coeffs[0, i] = 6 * interp_table[0, i] * tstep**3

        # coefficients have been mapped for PDQ interpolation, but the
        # higher order coefficients may be resolution limited. The data
        # can be bit-shifted on chip, and the shift is calculated here

        # number of significant bits for higher order terms
        sig_bits = np.log2(np.abs(new_coeffs[:-1, i]) + 1e-23)

        # number of shift bits is applied in multiples of N for Nth order coefficients, so
        # the overhead (or unused) bits are divided by the coefficient order. The shift is
        # applied to all non-zeroth order coefficients, so the maximum shift is determined
        # from the minimum overhead with multiples taken into account. The parameter width
        # is 40 bits, but is signed, so 39 sets the number of unsigned data bits. The shift
        # word is encoded in 5 bits, allowing for a maximum shift of 31. The data is written
        # to varying length words in fabric, where the word size is 40+16*N. However, if the
        # pulse time is long, more sensitivity is needed and the bit shift can exceed the
        # number of overhead bits when the coefficients are very small
        shift_len = int(
            min(min([(39 - v) / (i + 1) for i, v in enumerate(reversed(sig_bits))]), 31)
        )

        # re-map coefficients with bit shift, bit shift is applied as 2**(shift*N) as opposed
        # to simply bit-shifting the coefficients as in int(val)<<(shift*N). This is done in
        # order to get more resolution on the LSBs, which accumulate over many clock cycles
        new_coeffs[2, i] = int(new_coeffs[2, i] * (1 << shift_len))
        new_coeffs[1, i] = int(new_coeffs[1, i] * (1 << (shift_len * 2)))
        new_coeffs[0, i] = int(new_coeffs[0, i] * (1 << (shift_len * 3)))

        # shift_len can vary for every coefficient, so a list of shift parameters is passed
        # along with the coefficients for metadata tagging
        shift_len_list.append(shift_len)

    return new_coeffs, shift_len_list

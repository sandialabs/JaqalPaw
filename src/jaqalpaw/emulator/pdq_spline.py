def pdq_spline(coeffs, xdata, nsteps=409625, shift=0):
    """Generate spline data for mapped coefficients
    Note: nsteps = 1/tstep-1 used in cs_mapper"""
    final_data = []
    for n, x in enumerate(xdata):
        # Initialize coefficients for t0
        v0 = int(coeffs[3, n])
        v1 = int(coeffs[2, n])
        v2 = int(coeffs[1, n])
        v3 = int(coeffs[0, n])
        for i in range(nsteps):
            final_data.append(v0)
            v0 += v1 >> shift
            v1 += v2 >> shift
            v2 += v3 >> shift
    return final_data

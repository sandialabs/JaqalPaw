def pdq_spline(coeffs, xdata, nsteps=409625):
    """Generate spline data for mapped coefficients
    Note: nsteps = 1/tstep-1 used in cs_mapper"""
    final_data = []
    for n, x in enumerate(xdata):
        # Initialize coefficients for t0
        v0 = coeffs[3, n]
        v1 = coeffs[2, n]
        v2 = coeffs[1, n]
        v3 = coeffs[0, n]
        for i in range(nsteps - 1):
            final_data.append(v0)
            v0 += v1
            v1 += v2
            v2 += v3
    return final_data

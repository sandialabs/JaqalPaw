import numpy as np

#__all__ = ['cs_mapper', 'cs_mapper_int', 'pdq_spline', 'cs_mapper_int_auto_shift']

#def cs_mapper(interp_table, nsteps=409625):
    #"""Map spline coefficients into representation that
       #can be used for accumulator based reconstruction"""
    #tstep = 1/nsteps
    #new_coeffs = np.zeros(interp_table.shape)
    #for i in range(interp_table.shape[1]):
        #new_coeffs[3,i] = interp_table[3,i]
        #new_coeffs[2,i] = interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3
        #new_coeffs[1,i] = 2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3
        #new_coeffs[0,i] = 6*interp_table[0,i]*tstep**3
    #return new_coeffs

#def cs_mapper_int(interp_table, nsteps=409625, shift_len=16):
    #"""Map spline coefficients into representation that
       #can be used for accumulator based reconstruction.
       #Convert the data to integers for hardware and bitshift
       #higher order coefficients for better resolution"""
    #tstep = 1/nsteps
    #new_coeffs = np.zeros(interp_table.shape)
    #for i in range(interp_table.shape[1]):
        #new_coeffs[3,i] = int(interp_table[3,i])
        #new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
        #new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
        #new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
    #return new_coeffs

#def cs_mapper_int_auto_shift(interp_table, nsteps=409625):
    #"""Map spline coefficients into representation that
       #can be used for accumulator based reconstruction.
       #This variation automatically determines the optimal
       #bit shift for a given set of spline coefficients"""
    #tstep = 1/nsteps
    #new_coeffs = np.zeros(interp_table.shape)
    #shift_len_list = [0]*interp_table.shape[1]
    #for i in range(interp_table.shape[1]):
        ##try:
        #shift_len = 0
        ##for nn in range(2):
        #new_coeffs[3,i] = int(interp_table[3,i])
        #new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
        #new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
        #new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
            ##if nn == 0:
        #coeff_disp = [int(((39-np.log2(abs(new_coeffs[j,i])))/(3-j))) for j in range(3)]
        #shift_len = min(coeff_disp)
        #shift_len_list[i] = min(coeff_disp)#shift_len
        ##new_coeffs[3,i] = int(interp_table[3,i])
        #new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
        #new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
        #new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
        ###except Exception as e:
            ##raise e
    #return new_coeffs, shift_len_list
#
#def cs_mapper_int_auto_shift_orig(interp_table, nsteps=409625):
    #"""Map spline coefficients into representation that
       #can be used for accumulator based reconstruction.
       #This variation automatically determines the optimal
       #bit shift for a given set of spline coefficients"""
    #tstep = 1/nsteps
    #new_coeffs = np.zeros(interp_table.shape)
    #current_shift = 0
    #for shift_len in reversed(range(17)):
        #for i in range(interp_table.shape[1]):
            #try:
                #new_coeffs[3,i] = int(interp_table[3,i])
                #new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
                #new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
                #new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
                #for j in range(3):
                    #if new_coeffs[j,i] < 0:
                        #if len(bin(~int(new_coeffs[j,i])+1)) > 41:
                            #print(f"BROKEN AT SHIFT = {shift_len}")
                            #current_shift = shift_len-1
                            #break
                    ##elif len(bin(new_coeffs[j,i])) > 41:
                        #print(f"BROKEN AT SHIFT = {shift_len}")
                        ##current_shift = shift_len-1
                        #break
                #else:
                    #print(f"RETURNED AT SHIFT = {shift_len}")
            #except Exception as e:
                ##raise e
                #pass
    #shift_len = current_shift-1 if current_shift > 0 else 0
    #for i in range(interp_table.shape[1]):
        #new_coeffs[3,i] = int(interp_table[3,i])
        #new_coeffs[2,i] = int((interp_table[2,i]*tstep + interp_table[1,i]*tstep**2 + interp_table[0,i]*tstep**3)*2**shift_len)
        #new_coeffs[1,i] = int((2*interp_table[1,i]*tstep**2 + 6*interp_table[0,i]*tstep**3)*2**(shift_len*2))
        #new_coeffs[0,i] = int((6*interp_table[0,i]*tstep**3)*2**(shift_len*3))
    #return new_coeffs, shift_len

def pdq_spline(coeffs, xdata, nsteps=409625):
    """Generate spline data for mapped coefficients
       Note: nsteps = 1/tstep-1 used in cs_mapper"""
    final_data = []
    for n,x in enumerate(xdata):
        # Initialize coefficients for t0
        v0 = coeffs[3,n]
        v1 = coeffs[2,n]
        v2 = coeffs[1,n]
        v3 = coeffs[0,n]
        for i in range(nsteps-1):
            final_data.append(v0)
            v0 += v1
            v1 += v2
            v2 += v3
    return final_data

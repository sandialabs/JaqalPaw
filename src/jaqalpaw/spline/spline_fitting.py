from ctypes import CDLL, c_int, c_longlong, c_bool
import numpy as np
from pathlib import Path

# load the library
spline_path = Path(__file__).parent.absolute() / "spline.so"
mylib = CDLL(str(spline_path))

# C-type corresponding to numpy array
DATA_POINTER_TYPE = np.ctypeslib.ndpointer(dtype=np.float64,
                                      ndim=1,
                                      flags="C")

# C-type corresponding to numpy array
STEP_POINTER_TYPE = np.ctypeslib.ndpointer(dtype=np.int64,
                                      ndim=1,
                                      flags="C")

# define prototypes
mylib.FitSpline.argtypes = [DATA_POINTER_TYPE, STEP_POINTER_TYPE, c_int, c_bool]
mylib.FitSplineDistribute.argtypes = [DATA_POINTER_TYPE, c_int, c_int, c_bool]

def fit_spline(data, steplist, apply_phase_mask=False):
    mylib.FitSpline.restype = np.ctypeslib.ndpointer(dtype=c_longlong, shape=(len(data), 6))
    result = mylib.FitSpline(np.float64(data), np.int64(steplist), len(data), apply_phase_mask)
    fixedres = result.T[:,:-1]
    shift_len = fixedres[0,:]
    return np.float64(fixedres[2:,:]), shift_len.tolist()

def fit_spline_distribute(data, duration, apply_phase_mask=False):
    mylib.FitSplineDistribute.restype = np.ctypeslib.ndpointer(dtype=c_longlong, shape=(len(data), 6))
    result = mylib.FitSplineDistribute(np.float64(data), int(duration), len(data), apply_phase_mask)
    fixedres = result.T[:,:-1]
    shift_len = fixedres[5,:]
    durlist = fixedres[4,:]
    return np.float64(fixedres[3::-1,:]), shift_len.tolist(), durlist

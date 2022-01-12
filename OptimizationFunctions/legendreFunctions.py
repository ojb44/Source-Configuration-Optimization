#Needed to have a compileable legendre derivative, compatible with numba

import ctypes
from numba.extending import get_cython_function_address
from numba import njit

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0_1eval_legendre")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)
eval_legendre_float64_fn = functype(addr)

 
@njit
def numba_eval_legendre_float64(l,x):
    #gives the Legendre polynomial P_l(x)
    return eval_legendre_float64_fn(l, x)

@njit
def legendre_deriv(l,x,h): 
    #x can be array
    #uses central difference formula for find the derivative of P_l(x)
    #h should be a small float of order of 1e-5
    return (numba_eval_legendre_float64(l,x+h)-numba_eval_legendre_float64(l,x-h))/(2*h)

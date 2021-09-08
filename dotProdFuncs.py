from numba import njit
import numpy as np

@njit
def cosAngleBetween(t1,t2,p1,p2):
    return np.sin(t1)*np.cos(p1)*np.sin(t2)*np.cos(p2)+np.sin(t1)*np.sin(p1)*np.sin(t2)*np.sin(p2)+np.cos(t1)*np.cos(t2)

@njit
def derivThetaDotProd(ti,tk,pi,pk):
    return np.cos(ti)*np.cos(pi)*np.sin(tk)*np.cos(pk)+np.cos(ti)*np.sin(pi)*np.sin(tk)*np.sin(pk)-np.sin(ti)*np.cos(tk)

@njit
def derivPhiDotProd(ti,tk,pi,pk):
    return -np.sin(ti)*np.sin(pi)*np.sin(tk)*np.cos(pk)+np.sin(ti)*np.cos(pi)*np.sin(tk)*np.sin(pk)
from numba import njit
import numpy as np

@njit
def cosAngleBetween(t1,t2,p1,p2):
    #returns the dot product between two unit vectors defined by polar coordinates theta_1=t1, phi_1=p1, theta_2=t2, phi_2=p2
    return np.sin(t1)*np.cos(p1)*np.sin(t2)*np.cos(p2)+np.sin(t1)*np.sin(p1)*np.sin(t2)*np.sin(p2)+np.cos(t1)*np.cos(t2)


#next two functions are required for finding the gradients of sigma_rms^2
@njit
def derivThetaDotProd(ti,tk,pi,pk):
    #two unit vectors defined by polar coordinates theta_k=tk, phi_k=pk, theta_i=ti, phi_i=pi 
    #returns the dot product between the kth unit vector and the derivative of the ith with respect to theta_i
    return np.cos(ti)*np.cos(pi)*np.sin(tk)*np.cos(pk)+np.cos(ti)*np.sin(pi)*np.sin(tk)*np.sin(pk)-np.sin(ti)*np.cos(tk)

@njit
def derivPhiDotProd(ti,tk,pi,pk):
    #two unit vectors defined by polar coordinates theta_k=tk, phi_k=pk, theta_i=ti, phi_i=pi 
    #returns the dot product between the kth unit vector and the derivative of the ith with respect to phi_i
    return -np.sin(ti)*np.sin(pi)*np.sin(tk)*np.cos(pk)+np.sin(ti)*np.cos(pi)*np.sin(tk)*np.sin(pk)
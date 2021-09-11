from scipy.integrate import quad
import numpy as np
from scipy.special import eval_legendre

#Point sources

def pointaLIntegrand(x,l,d,a):
    """

    Parameters
    ----------
    x : float
        x=-cos(theta) in the integral
    l : int
        mode number.
    d : float
        distance of point source from centre of capsule.
    a : float
        radius of capsule.

    Returns
    -------
    float
        the integrand for finding the a_L terms for point sources.

    """
    return (d*x - a)/((d**2 + a**2 - 2*a*d*x)**(3/2))*eval_legendre(l,x)

def aL(l,d,a):
    """

    Parameters
    ----------
    l : int
        mode number.
    d : float
        distance of point source from centre of capsule.
    a : float
        radius of capsule.

    Returns
    -------
    float
        the term a_L in the Legendre expansion of the intensity on the capsule due to a single point source and distance d on a capsule of radius a.

    """
    aLNormFactor=0.5*quad(pointaLIntegrand, a/d,1,args=(0,d,a))[0]
    I=quad(pointaLIntegrand, a/d,1,args=(l,d,a))
    return ((2*l+1)/2)*I[0]/aLNormFactor

def aLValuesArray(d, a, numLTerms=50):
    """

    Parameters
    ----------
    d : float
        distance of point source from centre of capsule.
    a : float
        radius of capsule.
    numLTerms : int, optional
        number of terms. The default is 50.

    Returns
    -------
    array
        array of the a_L values for point sources.

    """
    return np.array([aL(l,d,a) for l in range(numLTerms)])

#a_L coefficients for d=4.84, a=1, as used in the paper
aLValues484=aLValuesArray(4.84,1,50)

#a_L coefficients for d=2.587, a=1, as used in the paper
aLValues2587=aLValuesArray(2.58733,1,50)


#Laser light

def laseraLIntegrand(x,l,etaPerp,beta,alpha):
    """

    Parameters
    ----------
    x : float
        x=-cos(theta) in the integral
    l : int
        mode number.
    etaPerp : float
        parameter describing absorption efficiency
    beta : float
        parameter describing shape of beam
    alpha : float
        parameter describing shape of beam

    Returns
    -------
    float
        the integrand for finding the a_L terms for super-Gaussian profile lasers, accounting for non-perfect absorption.

    """
    return (1-(1-etaPerp)**(x**3))*np.exp(-(((1-x**2)/(alpha**2))**(beta/2)))*x*eval_legendre(l,x)

def laseraL(l,etaPerp,beta,alpha):
    """

    Parameters
    ----------
    l : int
        mode number.
    etaPerp : float
        parameter describing absorption efficiency
    beta : float
        parameter describing shape of beam
    alpha : float
        parameter describing shape of beam

    Returns
    -------
    float
        the a_L term for super-Gaussian profile lasers, accounting for non-perfect absorption.

    """
    aLNormFactor=0.5*quad(laseraLIntegrand,0,1,args=(0,etaPerp,beta,alpha))[0]
    I=quad(laseraLIntegrand,0,1,args=(l,etaPerp,beta,alpha))
    return ((2*l+1)/2)*I[0]/aLNormFactor

def aLValuesLaserArray(etaPerp,beta,alpha, numLTerms=50):
    """

    Parameters
    ----------
    etaPerp : float
        parameter describing absorption efficiency
    beta : float
        parameter describing shape of beam
    alpha : float
        parameter describing shape of beam
    numLTerms : int, optional
        number of terms. The default is 50.

    Returns
    -------
    array
        array of the a_L values for lasers.

    """
    return np.array([laseraL(l,etaPerp,beta,alpha) for l in range(numLTerms)])

#a_L coefficients for the case etaPerp=0.9, beta=5, alpha=0.95
aLValuesLaser=aLValuesLaserArray(0.9,5,0.95)
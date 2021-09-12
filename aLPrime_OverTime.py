#over time a_l values
import numpy as np
from aLValueFuncs import pointaLIntegrand, laseraLIntegrand
from scipy import integrate

#Point sources

def bVal(l,d,a):
    #b is the inner integral for a_l', defined in the paper
    I=integrate.quad(pointaLIntegrand,a/d,1,args=(l,d,a))
    return ((2*l+1)/2)*I[0]

def aPrimeLVal(l,d,tVals,rVals,T):
    """

    Parameters
    ----------
    l : int
        mode number.
    d : float
        distance of source from capsule.
    tVals : array of floats - must have size 2^k+1 
        time values.
    rVals : array of floats
        radius of the capsule at times corresponding to tVals.
    T : float
        total time.

    Returns
    -------
    float
        return a_l', the effective a_l value for time averaging, for point sources.

    """
    #the arrays must have 2^k + 1 values in them for romb to work
    numSamples=tVals.size
    bArray=np.zeros(numSamples)
    for i in range(numSamples):
        bArray[i]=bVal(l,d,rVals[i])
    I=integrate.romb(bArray,dx=T/(numSamples-1))
    
    b0Array=np.zeros(numSamples)
    for i in range(numSamples):
        b0Array[i]=bVal(0,d,rVals[i])
    #note that the Ts cancel in next two lines, so are not included
    #print(b0Array)
    iTBar=integrate.romb(b0Array,dx=T/(numSamples-1))
    return I/iTBar

def createALPrimeVals(d,tVals,rVals,T,numLTerms=50):
    return np.array([aPrimeLVal(l,d,tVals,rVals,T) for l in range(numLTerms)])


#Lasers

def bValLaser(l,etaPerp,beta,alpha):
    #b is the inner integral for a_l', defined in the paper
    I=integrate.quad(laseraLIntegrand,0,1,args=(0,etaPerp,beta,alpha))
    return ((2*l+1)/2)*I[0]

def aPrimeLValLaser(l,etaPerp,beta,alpha_0,tVals,rVals,T):
    """

    Parameters
    ----------
    l : int
        mode number.
    etaPerp : float
        parameter describing absorption efficiency
    beta : float
        parameter describing shape of beam
    alpha_0 : float
        the INITIAL value of alpha - this will change as the capsule radius changes
    tVals : array of floats - must have size 2^k+1 
        time values.
    rVals : array of floats
        radius of the capsule at times corresponding to tVals.
    T : float
        total time.

    Returns
    -------
    float
        return a_l', the effective a_l value for time averaging, for point sources.

    """
    #the arrays must have 2^k + 1 values in them for romb to work
    numSamples=tVals.size
    bArray=np.zeros(numSamples)
    for i in range(numSamples):
        alpha=alpha_0*(rVals[0]/rVals[i])  #the alpha value changes depending on the radius
        bArray[i]=bValLaser(l,etaPerp,beta,alpha)
    I=integrate.romb(bArray,dx=T/(numSamples-1))
    
    b0Array=np.zeros(numSamples)
    for i in range(numSamples):
        alpha=alpha_0*(rVals[0]/rVals[i])
        b0Array[i]=bValLaser(0,etaPerp,beta,alpha)
    #note that the Ts cancel in next two lines, so are not included
    #print(b0Array)
    iTBar=integrate.romb(b0Array,dx=T/(numSamples-1))
    return I/iTBar

def createALPrimeValsLaser(etaPerp,beta,alpha_0,tVals,rVals,T,numLTerms=50):
    return np.array([aPrimeLValLaser(l,etaPerp,beta,alpha_0,tVals,rVals,T) for l in range(numLTerms)])


#example data used in paper
tValsExample=np.linspace(0,1,2**4+1)
rValsExample=np.array([1.       , 1.       , 1.       , 1.00016  , 0.998495 , 0.99057  ,
       0.978295 , 0.96078  , 0.93875  , 0.90171  , 0.855955 , 0.78125  ,
       0.698475 , 0.579015 , 0.436411 , 0.2602765, 0.065    ])

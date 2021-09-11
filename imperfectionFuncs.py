#Imperfection functions:
    
from numba import njit
import numpy as np
import sigmaRMSSquFuncs


@njit
def generatePerturbationsScaled(thetaStd,phiStd,strStd,numSources,numPerts,configVector):
    """

    Parameters
    ----------
    stdTheta : float
        standard deviation in theta values for the perturbations.
    stdPhi : float
        standard deviation in phi values for the perturbations.
    stdStr : float
        standard deviation in strength values for the perturbations - defined as fraction of the mean strength.
    numSources : TYPE
        DESCRIPTION.
    numPerts : int
        number of perturbed configuration vectors to generate.
    configVector : array
        array describing the configuration - in order of all theta values, then phi, then strengths.

    Returns
    -------
    array of arrays
        array of numPerts new configuration vectors, which are all small perturbations from the original.

    """
    thetaPerts=np.random.normal(0,thetaStd,(numPerts,numSources))
    phiPerts=np.random.normal(0,phiStd,(numPerts,numSources))
    meanStr=np.mean(configVector[2*numSources:])
    strPerts=np.random.normal(0,strStd,(numPerts,numSources))*meanStr
    
    return np.concatenate((thetaPerts,phiPerts,strPerts),axis=1)
    
@njit
def sigmaRMSAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerts):
    """

    Parameters
    ----------
    configVector : array
        array describing the configuration - in order of all theta values, then phi, then strengths.
    perturbations: array of arrays
        array of the new configuration vectors, which are all small perturbations from the original.
    aLValues : array of floats
        the coefficients a_L in the Legendre expansion of intensity due to a single source.
    numSources : int
        number of sources.
    numLTerms : int
        number of terms in Legendre expansion.
    numPerts : int
        number of perturbed configuration vectors to generate.

    Returns
    -------
    sigmaRMSVals : float
        array of the sigma_rms values of all the configurations in the array perturbations.

    """
    #this one uses sigma rather than sigma^2
    configVectors=perturbations+configVector
    sigmaRMSVals=np.zeros(numPerts)
    for i in range(numPerts):
        sigmaRMSVals[i]=np.sqrt(sigmaRMSSquFuncs.sigmaRMSSqu(configVectors[i],numSources,aLValues,numLTerms))
    return sigmaRMSVals

@njit
def stabilityTest(configVector,numSources,aLValues,numLTerms,numPerts,stdTheta,stdPhi,stdStr):
    """

    Parameters
    ----------
    configVector : array
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    aLValues : array of floats
        the coefficients a_L in the Legendre expansion of intensity due to a single source.
    numLTerms : int
        number of terms in Legendre expansion.
    numPerts : int
        number of perturbed configuration vectors to generate.
    stdTheta : float
        standard deviation in theta values for the perturbations.
    stdPhi : float
        standard deviation in phi values for the perturbations.
    stdStr : float
        standard deviation in strength values for the perturbations - defined as fraction of the mean strength.

    Returns
    -------
    float
        the ratio of the mean sigma_rms of all the perturbed configuration vectors to the absolute minimum sigma_rms.
    float
        the standard deviation of sigma_rms of all the perturbed configuration vectors.

    """
    #this one uses sigma rather than sigma^2
    perts=generatePerturbationsScaled(stdTheta,stdPhi,stdStr,numSources,numPerts,configVector)
    sigmaRMSVals=sigmaRMSAfterPerts(configVector, perts, aLValues, numSources, numLTerms, numPerts)
    mean=sigmaRMSVals.mean()
    std=sigmaRMSVals.std()
    minVal=np.sqrt(sigmaRMSSquFuncs.sigmaRMSSqu(configVector,numSources,aLValues,numLTerms))
    return mean/minVal,std/minVal   #note that this has changed from the original definition I had
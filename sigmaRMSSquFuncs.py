#Functions to calculate sigmaRMSSqu and sigmaRMSSquGrad

import legendreFunctions
from dotProdFuncs import cosAngleBetween, derivThetaDotProd, derivPhiDotProd
import numpy as np
from numba import njit


@njit
def gLSqu(l, configVector, numSources):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : array
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.

    Returns
    -------
    float
        G_l^2, as defined in the paper.

    """
    sumGrid=np.zeros((numSources,numSources))
    for i in range(numSources):
        for j in range(numSources):
            sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]))       
    
    strengthArr=configVector[2*numSources:]
    strengthGrid=np.outer(strengthArr, strengthArr)
    
    sumGrid=sumGrid*strengthGrid
    
    normStrength=np.sum(strengthArr)**2
    
    return np.sum(sumGrid)/normStrength



@njit
def sigmaLSqu(l,configVector,numSources,aLValues):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : array 
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    aLValues : array
        the coefficients a_L in the Legendre expansion of intensity due to a single source.

    Returns
    -------
    float
        sigma_l ^ 2 as defined in the paper.

    """
    aLVal=aLValues[l]
    gLVal=gLSqu(l,configVector,numSources)
    return (aLVal**2 * gLVal) / (2*l+1)


@njit
def sigmaRMSSqu(configVector,numSources,aLValues,numLTerms):
    """

    Parameters
    ----------
    configVector : array
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    aLValues : array
        the coefficients a_L in the Legendre expansion of intensity due to a single source.
    numLTerms : int
        number of terms kept in Legendre expansion.

    Returns
    -------
    float
        sigma_{rms}^2, the rms squared nonuniformity in intensity.

    """
    sigmaLVals=np.arange(1.0, numLTerms+1)
    for i in range(numLTerms):
        sigmaLVals[i]=sigmaLSqu(i+1,configVector,numSources,aLValues)
    return np.sum(sigmaLVals)

@njit
def gLSquGradS(l, configVector, numSources, sourceIndex):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : array 
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    sourceIndex : int
        index of source which is being varied
    Returns
    -------
    float
        the partial derivative of G_l^2 with respect to the strength of the source labelled by sourceIndex

    """
    sumGrid=np.zeros((numSources,numSources))
    for i in range(numSources):
        for j in range(numSources):
            #sumGrid[i][j]=cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
            sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]))       
    
    
    strengthArr=configVector[2*numSources:]
    strengthGrid=np.outer(strengthArr, strengthArr)
    
    
    sumGrid=sumGrid*strengthGrid
    
    firstSumTerm=np.sum(sumGrid[sourceIndex])/(strengthArr[sourceIndex])
    secondSumTerm=np.sum(sumGrid)
    
    IT=np.sum(strengthArr)
    
    return 2*firstSumTerm/(IT**2) -(2/(IT)**3)*secondSumTerm


@njit
def gLSquGradTheta(l, configVector, numSources, sourceIndex):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : array 
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    sourceIndex : int
        index of source which is being varied
    Returns
    -------
    float
        the partial derivative of G_l^2 with respect to the theta position of the source labelled by sourceIndex

    """
    
    sumArr=np.zeros(numSources)
    i=sourceIndex
    for j in range(numSources):
        #sumGrid[i][j]=cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
        sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]),0.00001)
    # poly=legendre(l)
    # polyd=poly.deriv()
    # sumArr=np.polyval(polyd,sumArr)
    
    strengthArr=configVector[2*numSources:]
    IT=np.sum(strengthArr)
        
    sumArr2=np.zeros(numSources)
    for j in range(numSources):
        #sumArr2[j]=configVector[2*numSources+i]*configVector[2*numSources+j]*derivThetaDotProd(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
        sumArr2[j]=derivThetaDotProd(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
    
    return 2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2)

#*strengthArr*strengthArr[i]

@njit
def gLSquGradPhi(l, configVector, numSources, sourceIndex):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : array 
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    sourceIndex : int
        index of source which is being varied
    Returns
    -------
    float
        the partial derivative of G_l^2 with respect to the phi position of the source labelled by sourceIndex

    """
    sumArr=np.zeros(numSources)
    i=sourceIndex
    for j in range(numSources):
        #sumGrid[i][j]=cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
        sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]),0.00001)
    # poly=legendre(l)
    # polyd=poly.deriv()
    # sumArr=np.polyval(polyd,sumArr)
    
    strengthArr=configVector[2*numSources:]
    IT=np.sum(strengthArr)
    
    sumArr2=np.zeros(numSources)
    for j in range(numSources):
        sumArr2[j]=derivPhiDotProd(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
    
    return 2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2)

@njit
def sigmaLSquGrad(l,configVector,numSources,aLValues):
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : TYPE
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    aLValues : array of floats
        the coefficients a_L in the Legendre expansion of intensity due to a single source.

    Returns
    -------
    derivArr : array of floats
        the gradient vector of sigma_l^2 in the space of the configuration vector.

    """
    derivArr=np.zeros(3*numSources)
    for i in range(numSources):
        derivArr[i]=gLSquGradTheta(l, configVector, numSources, i)
        derivArr[i+numSources]=gLSquGradPhi(l, configVector, numSources, i)
        derivArr[i+2*numSources]=gLSquGradS(l, configVector, numSources, i)
        #above is removed such that gradient is zero for strengths
    aLVal=aLValues[l]
    derivArr=derivArr*(aLVal**2)/(2*l+1)
    return derivArr

@njit
def sigmaRMSSquGrad(configVector,numSources,aLValues,numLTerms):  
    """

    Parameters
    ----------
    l : int
        mode number.
    configVector : TYPE
        array describing the configuration - in order of all theta values, then phi, then strengths.
    numSources : int
        number of sources.
    aLValues : array of floats
        the coefficients a_L in the Legendre expansion of intensity due to a single source.

    Returns
    -------
    gradArr : array of floats
        the gradient vector of sigma_{rms}^2 in the space of the configuration vector.

    """
    gradArr=np.zeros(3*numSources)
    for l in range(numLTerms):
        gradArr+=sigmaLSquGrad(l,configVector,numSources,aLValues)
    return gradArr
#reduced degrees of freedom

from numba import njit
import numpy as np
import legendreFunctions
from dotProdFuncs import cosAngleBetween, derivThetaDotProd, derivPhiDotProd

#the following functions are identical to the functions in the file sigmaRMSSqu, but account for having only half the configuration vector (other points reflected through the origin), and fix one source at (0,0), and one with phi=0, so that configurations are now unique

@njit
def gLSquRDOF(l, configVector, halfNumSources):
    
    if l%2==0:
        sumGrid=np.zeros((halfNumSources,halfNumSources))
        for i in range(halfNumSources):
            for j in range(halfNumSources):
                sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources]))       
        strengthArr=configVector[2*halfNumSources:]
        strengthGrid=np.outer(strengthArr, strengthArr)
        sumGrid=sumGrid*strengthGrid
        normStrength=(2*np.sum(strengthArr))**2
        return np.sum(sumGrid)/normStrength*4
    
    else:
        return 0

@njit
def sigmaLSquRDOF(l,configVector,halfNumSources,aLValues):
    aLVal=aLValues[l]
    gLVal=gLSquRDOF(l,configVector,halfNumSources)
    return (aLVal**2 * gLVal) / (2*l+1)

@njit
def sigmaRMSSquRDOF(configVector,halfNumSources,aLValues,numLTerms):
    sigmaLVals=np.arange(1.0, numLTerms+1)
    for i in range(numLTerms):
        sigmaLVals[i]=sigmaLSquRDOF(i+1,configVector,halfNumSources,aLValues)
    return np.sum(sigmaLVals)

@njit
def gLSquGradSRDOF(l, configVector, halfNumSources, sourceIndex):
    if sourceIndex==0:
        return 0
    
    elif l%2==0:
        sumGrid=np.zeros((halfNumSources,halfNumSources))
        for i in range(halfNumSources):
            for j in range(halfNumSources):
                sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources]))       
        strengthArr=configVector[2*halfNumSources:]
        strengthGrid=np.outer(strengthArr, strengthArr)
        sumGrid=sumGrid*strengthGrid
        firstSumTerm=np.sum(sumGrid[sourceIndex])/(strengthArr[sourceIndex])*2
        secondSumTerm=np.sum(sumGrid)*4
        IT=np.sum(strengthArr)*2 #*2 accounts for only having half configVector
        return (2*firstSumTerm/(IT**2)-(2/(IT)**3)*secondSumTerm) *2  
    
    else:
        return 0

@njit
def gLSquGradThetaRDOF(l, configVector, halfNumSources, sourceIndex):
    if sourceIndex==0:
        return 0
    elif l%2==0:
        sumArr=np.zeros(halfNumSources)
        i=sourceIndex
        for j in range(halfNumSources):
            sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources]),0.00001)
        strengthArr=configVector[2*halfNumSources:]
        IT=np.sum(strengthArr)*2

        sumArr2=np.zeros(halfNumSources)
        for j in range(halfNumSources):
            sumArr2[j]=derivThetaDotProd(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources])

        return (2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2))*2*2
    
    else:
        return 0


@njit
def gLSquGradPhiRDOF(l, configVector, halfNumSources, sourceIndex):
    if sourceIndex==0 or sourceIndex==1:  #don't change phi of these two positions to maintain an orientation
        return 0
    elif l%2==0:
        sumArr=np.zeros(halfNumSources)
        i=sourceIndex
        for j in range(halfNumSources):
            sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources]),0.00001)
        strengthArr=configVector[2*halfNumSources:]
        IT=np.sum(strengthArr)*2
        sumArr2=np.zeros(halfNumSources)
        for j in range(halfNumSources):
            sumArr2[j]=derivPhiDotProd(configVector[i],configVector[j],configVector[i+halfNumSources],configVector[j+halfNumSources])
        return (2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2))*2*2
    
    else:
        return 0

@njit
def sigmaLSquGradRDOF(l,configVector,halfNumSources,aLValues):
    derivArr=np.zeros(3*halfNumSources)
    for i in range(halfNumSources):
        derivArr[i]=gLSquGradThetaRDOF(l, configVector, halfNumSources, i)
        derivArr[i+halfNumSources]=gLSquGradPhiRDOF(l, configVector, halfNumSources, i)
        derivArr[i+2*halfNumSources]=gLSquGradSRDOF(l, configVector, halfNumSources, i)
    aLVal=aLValues[l]
    derivArr=derivArr*(aLVal**2)/(2*l+1)
    return derivArr

@njit
def sigmaRMSSquGradRDOF(configVector,halfNumSources,aLValues,numLTerms):    
    gradArr=np.zeros(3*halfNumSources)
    for l in range(numLTerms):
        gradArr+=sigmaLSquGradRDOF(l,configVector,halfNumSources,aLValues)
    return gradArr

@njit
def sigmaRMSFuncRDOF(configVector,halfNumSources,aLValues,numLTerms):
    srs=sigmaRMSSquRDOF(configVector,halfNumSources,aLValues,numLTerms)
    srsg=sigmaRMSSquGradRDOF(configVector,halfNumSources,aLValues,numLTerms)
    return srs, srsg



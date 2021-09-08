#Functions to calculate sigmaRMSSqu and sigmaRMSSquGrad

import legendreFunctions
import numpy as np
from numba import njit

@njit
def cosAngleBetween(t1,t2,p1,p2):
    return np.sin(t1)*np.cos(p1)*np.sin(t2)*np.cos(p2)+np.sin(t1)*np.sin(p1)*np.sin(t2)*np.sin(p2)+np.cos(t1)*np.cos(t2)


@njit
def gLSqu(l, configVector, numSources):
    sumGrid=np.zeros((numSources,numSources))
    for i in range(numSources):
        for j in range(numSources):
            #sumGrid[i][j]=cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
            sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]))       
    
    #legArr=np.array([numba_eval_legendre_float64(val,l) for val in arr])
    #sumGrid=np.array([])
    #poly=legendre(l)
    #sumGrid=np.polyval(poly,sumGrid)
    
    strengthArr=configVector[2*numSources:]
    strengthGrid=np.outer(strengthArr, strengthArr)
    
    
    sumGrid=sumGrid*strengthGrid
    
    normStrength=np.sum(strengthArr)**2
    
    return np.sum(sumGrid)/normStrength



@njit
def sigmaLSqu(l,configVector,numSources,aLValues):
    aLVal=aLValues[l]
    gLVal=gLSqu(l,configVector,numSources)
    return (aLVal**2 * gLVal) / (2*l+1)


@njit
def sigmaRMSSqu(configVector,numSources,aLValues,numLTerms):
    sigmaLVals=np.arange(1.0, numLTerms+1)
    for i in range(numLTerms):
        sigmaLVals[i]=sigmaLSqu(i+1,configVector,numSources,aLValues)
    return np.sum(sigmaLVals)

@njit
def gLSquGradS(l, configVector, numSources, sourceIndex):
    sumGrid=np.zeros((numSources,numSources))
    for i in range(numSources):
        for j in range(numSources):
            #sumGrid[i][j]=cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources])
            sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(configVector[i],configVector[j],configVector[i+numSources],configVector[j+numSources]))       
    
    #legArr=np.array([numba_eval_legendre_float64(val,l) for val in arr])
    #sumGrid=np.array([])
    #poly=legendre(l)
    #sumGrid=np.polyval(poly,sumGrid)
    
    strengthArr=configVector[2*numSources:]
    strengthGrid=np.outer(strengthArr, strengthArr)
    
    
    sumGrid=sumGrid*strengthGrid
    
    firstSumTerm=np.sum(sumGrid[sourceIndex])/(strengthArr[sourceIndex])
    secondSumTerm=np.sum(sumGrid)
    
    IT=np.sum(strengthArr)
    
    return 2*firstSumTerm/(IT**2) -(2/(IT)**3)*secondSumTerm

@njit
def derivThetaDotProd(ti,tk,pi,pk):
    return np.cos(ti)*np.cos(pi)*np.sin(tk)*np.cos(pk)+np.cos(ti)*np.sin(pi)*np.sin(tk)*np.sin(pk)-np.sin(ti)*np.cos(tk)

@njit
def derivPhiDotProd(ti,tk,pi,pk):
    return -np.sin(ti)*np.sin(pi)*np.sin(tk)*np.cos(pk)+np.sin(ti)*np.cos(pi)*np.sin(tk)*np.sin(pk)

@njit
def gLSquGradTheta(l, configVector, numSources, sourceIndex):
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
    gradArr=np.zeros(3*numSources)
    for l in range(numLTerms):
        gradArr+=sigmaLSquGrad(l,configVector,numSources,aLValues)
    return gradArr
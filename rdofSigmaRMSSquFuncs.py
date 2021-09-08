#reduced degrees of freedom

from numba import njit
import numpy as np
import legendreFunctions
from dotProdFuncs import cosAngleBetween, derivThetaDotProd, derivPhiDotProd

@njit
def gLSquRDOF(l, chromosome, halfNumSources):
    
    if l%2==0:
        
        sumGrid=np.zeros((halfNumSources,halfNumSources))
        
        for i in range(halfNumSources):
            for j in range(halfNumSources):
                #sumGrid[i][j]=cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
                sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources]))       
        
        #legArr=np.array([numba_eval_legendre_float64(val,l) for val in arr])
        #sumGrid=np.array([])
        #poly=legendre(l)
        #sumGrid=np.polyval(poly,sumGrid)
        
        strengthArr=chromosome[2*halfNumSources:]
        strengthGrid=np.outer(strengthArr, strengthArr)


        sumGrid=sumGrid*strengthGrid

        normStrength=(2*np.sum(strengthArr))**2

        return np.sum(sumGrid)/normStrength  *4
    
    else:
        return 0

@njit
def sigmaLSquRDOF(l,chromosome,halfNumSources,aLValues):
    aLVal=aLValues[l]
    gLVal=gLSquRDOF(l,chromosome,halfNumSources)
    return (aLVal**2 * gLVal) / (2*l+1)

@njit
def sigmaRMSSquRDOF(chromosome,halfNumSources,aLValues,numLTerms):
    sigmaLVals=np.arange(1.0, numLTerms+1)
    for i in range(numLTerms):
        sigmaLVals[i]=sigmaLSquRDOF(i+1,chromosome,halfNumSources,aLValues)
    return np.sum(sigmaLVals)

@njit
def gLSquGradSRDOF(l, chromosome, halfNumSources, sourceIndex):
    if sourceIndex==0:
        return 0
    
    elif l%2==0:
        sumGrid=np.zeros((halfNumSources,halfNumSources))
        for i in range(halfNumSources):
            for j in range(halfNumSources):
                #sumGrid[i][j]=cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
                sumGrid[i][j]=legendreFunctions.numba_eval_legendre_float64(l,cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources]))       

        #legArr=np.array([numba_eval_legendre_float64(val,l) for val in arr])
        #sumGrid=np.array([])
        #poly=legendre(l)
        #sumGrid=np.polyval(poly,sumGrid)

        strengthArr=chromosome[2*halfNumSources:]
        strengthGrid=np.outer(strengthArr, strengthArr)


        sumGrid=sumGrid*strengthGrid

        firstSumTerm=np.sum(sumGrid[sourceIndex])/(strengthArr[sourceIndex])*2
        secondSumTerm=np.sum(sumGrid)*4

        IT=np.sum(strengthArr)*2 #*2 accounts for only having half chromosome

        return (2*firstSumTerm/(IT**2)-(2/(IT)**3)*secondSumTerm) *2  
    
    else:
        return 0

@njit
def gLSquGradThetaRDOF(l, chromosome, halfNumSources, sourceIndex):
    if sourceIndex==0:
        return 0
    elif l%2==0:
        sumArr=np.zeros(halfNumSources)
        i=sourceIndex
        for j in range(halfNumSources):
            #sumGrid[i][j]=cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
            sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources]),0.00001)
        # poly=legendre(l)
        # polyd=poly.deriv()
        # sumArr=np.polyval(polyd,sumArr)

        strengthArr=chromosome[2*halfNumSources:]
        IT=np.sum(strengthArr)*2

        sumArr2=np.zeros(halfNumSources)
        for j in range(halfNumSources):
            #sumArr2[j]=chromosome[2*numSources+i]*chromosome[2*numSources+j]*derivThetaDotProd(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
            sumArr2[j]=derivThetaDotProd(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources])

        return (2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2))*2*2
    
    else:
        return 0


@njit
def gLSquGradPhiRDOF(l, chromosome, halfNumSources, sourceIndex):
    if sourceIndex==0 or sourceIndex==1:  #don't change phi of these two positions to maintain an orientation
        return 0
    elif l%2==0:
        sumArr=np.zeros(halfNumSources)
        i=sourceIndex
        for j in range(halfNumSources):
            #sumGrid[i][j]=cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
            sumArr[j]=legendreFunctions.legendre_deriv(l,cosAngleBetween(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources]),0.00001)
        # poly=legendre(l)
        # polyd=poly.deriv()
        # sumArr=np.polyval(polyd,sumArr)

        strengthArr=chromosome[2*halfNumSources:]
        IT=np.sum(strengthArr)*2

        sumArr2=np.zeros(halfNumSources)
        for j in range(halfNumSources):
            #sumArr2[j]=chromosome[2*numSources+i]*chromosome[2*numSources+j]*derivThetaDotProd(chromosome[i],chromosome[j],chromosome[i+numSources],chromosome[j+numSources])
            sumArr2[j]=derivPhiDotProd(chromosome[i],chromosome[j],chromosome[i+halfNumSources],chromosome[j+halfNumSources])

        return (2*np.sum(sumArr*sumArr2*strengthArr*strengthArr[i])/(IT**2))*2*2
    
    else:
        return 0

@njit
def sigmaLSquGradRDOF(l,chromosome,halfNumSources,aLValues):
    derivArr=np.zeros(3*halfNumSources)
    for i in range(halfNumSources):
        derivArr[i]=gLSquGradThetaRDOF(l, chromosome, halfNumSources, i)
        derivArr[i+halfNumSources]=gLSquGradPhiRDOF(l, chromosome, halfNumSources, i)
        derivArr[i+2*halfNumSources]=gLSquGradSRDOF(l, chromosome, halfNumSources, i)
        #above is removed such that gradient is zero for strengths
    aLVal=aLValues[l]
    derivArr=derivArr*(aLVal**2)/(2*l+1)
    return derivArr

@njit
def sigmaRMSSquGradRDOF(chromosome,halfNumSources,aLValues,numLTerms):    
    gradArr=np.zeros(3*halfNumSources)
    for l in range(numLTerms):
        gradArr+=sigmaLSquGradRDOF(l,chromosome,halfNumSources,aLValues)
    return gradArr

@njit
def sigmaRMSFuncRDOF(chromosome,halfNumSources,aLValues,numLTerms):
    srs=sigmaRMSSquRDOF(chromosome,halfNumSources,aLValues,numLTerms)
    srsg=sigmaRMSSquGradRDOF(chromosome,halfNumSources,aLValues,numLTerms)
    return srs, srsg



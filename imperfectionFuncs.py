#Imperfection functions:
    
from numba import njit
import numpy as np
import sigmaRMSSquFuncs

@njit
def generatePerturbations(thetaStd,phiStd,strStd,numSources,numPerturbations):
    thetaPerts=np.random.normal(0,thetaStd,(numPerturbations,numSources))
    phiPerts=np.random.normal(0,phiStd,(numPerturbations,numSources))
    strPerts=np.random.normal(0,strStd,(numPerturbations,numSources))
    
    return np.concatenate((thetaPerts,phiPerts,strPerts),axis=1)
    
@njit
def sigmaRMSSquAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations):
    configVectors=perturbations+configVector
    sigmaRMSVals=np.zeros(numPerturbations)
    for i in range(numPerturbations):
        sigmaRMSVals[i]=sigmaRMSSquFuncs.sigmaRMSSqu(configVectors[i],numSources,aLValues,numLTerms)
    return sigmaRMSVals

@njit
def meanAndStdAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations):
    sigmaRMSVals=sigmaRMSSquAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations)
    return sigmaRMSVals.mean(),sigmaRMSVals.std()

@njit
def meanDiffAndStdAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations):
    sigmaRMSVals=sigmaRMSSquAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations)
    return sigmaRMSVals.mean()-sigmaRMSSquFuncs.sigmaRMSSqu(configVector,numSources,aLValues,numLTerms),sigmaRMSVals.std()

@njit
def generatePerturbationsScaled(thetaStd,phiStd,strStd,numSources,numPerturbations,configVector):
    thetaPerts=np.random.normal(0,thetaStd,(numPerturbations,numSources))
    phiPerts=np.random.normal(0,phiStd,(numPerturbations,numSources))
    meanStr=np.mean(configVector[2*numSources:])
    strPerts=np.random.normal(0,strStd,(numPerturbations,numSources))*meanStr
    
    return np.concatenate((thetaPerts,phiPerts,strPerts),axis=1)
    

@njit
def stabilityTestSqu(configVector,aLValues,numSources,numLTerms,numPerts,stdTheta,stdPhi,stdStr):
    #returns the (average - min) / min of the results from all the perturbations
    #strength perturbations are done in proportion to the mean strength
    perts=generatePerturbationsScaled(stdTheta,stdPhi,stdStr,numSources,numPerts,configVector)
    meanDiffAndStd=meanDiffAndStdAfterPerts(configVector,perts,aLValues,numSources,numLTerms,numPerts)
    return (meanDiffAndStd[0]/sigmaRMSSquFuncs.sigmaRMSSqu(configVector,numSources,aLValues,numLTerms)), meanDiffAndStd[1]

@njit
def sigmaRMSAfterPerts(configVector, perturbations, aLValues, numSources, numLTerms, numPerturbations):
    #this one uses sigma rather than sigma^2
    configVectors=perturbations+configVector
    sigmaRMSVals=np.zeros(numPerturbations)
    for i in range(numPerturbations):
        sigmaRMSVals[i]=np.sqrt(sigmaRMSSquFuncs.sigmaRMSSqu(configVectors[i],numSources,aLValues,numLTerms))
    return sigmaRMSVals

@njit
def stabilityTest(configVector,numSources,aLValues,numLTerms,numPerts,stdTheta,stdPhi,stdStr):
    #this one uses sigma rather than sigma^2
    perts=generatePerturbationsScaled(stdTheta,stdPhi,stdStr,numSources,numPerts,configVector)
    sigmaRMSVals=sigmaRMSAfterPerts(configVector, perts, aLValues, numSources, numLTerms, numPerts)
    mean=sigmaRMSVals.mean()
    std=sigmaRMSVals.std()
    minVal=np.sqrt(sigmaRMSSquFuncs.sigmaRMSSqu(configVector,numSources,aLValues,numLTerms))
    return mean/minVal,std/minVal   #note that this has changed from the original definition I had
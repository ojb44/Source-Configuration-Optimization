import sigmaRMSSquFuncs
import customOptimizerSettings
import aLValueFuncs
import imperfectionFuncs
import numpy as np
from scipy.optimize import basinhopping

class SourceType:
    def __init__(self, aLVals):
        self.aLVals=aLVals
        
class SingleSource:
    def __init__(self, theta, phi, strength):
        self.theta=theta
        self.phi=phi
        self.strength=strength
        
    def sourceTuple(self):
        return (self.theta,self.phi,self.strength)
        
class Configuration:
    def __init__(self, numSources, sourceType, sourceArray):
        #self.sourceArray=sourceArray
        self.numSources=numSources
        self.sourceType=sourceType
        self.configVector=self.configVectorFromSourceArray(sourceArray)
        
    def configVectorFromSourceArray(self,sourceArray):
        thetas=list(map(lambda x: x.theta, sourceArray))
        phis=list(map(lambda x: x.phi, sourceArray))
        strengths=list(map(lambda x: x.strength, sourceArray))
        return np.array([thetas,phis,strengths]).flatten()
    
    def sourceArrayFromConfigVector(self):
        sourceArray=np.empty(self.numSources,dtype=SingleSource)
        for i in range(self.numSources):
            sourceArray[i]=SingleSource(self.configVector[i],self.configVector[i+self.numSources],self.configVector[i+2*self.numSources])
        return sourceArray
    
    def sourceTuples(self):
        sourceArray=self.sourceArrayFromConfigVector()
        return list(map(lambda x: x.sourceTuple(),sourceArray))
    
    def sigmaRMSSqu(self,numLTerms):
        return sigmaRMSSquFuncs.sigmaRMSSqu(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms)

    def sigmaRMSSquGrad(self,numLTerms):
        return sigmaRMSSquFuncs.sigmaRMSSquGrad(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms)

    def stabilityTest(self,numLTerms,numPerturbations,stdTheta,stdPhi,stdStr):
        return imperfectionFuncs.stabilityTest(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms,numPerturbations,stdTheta,stdPhi,stdStr)

pointSource484=SourceType(aLValueFuncs.aLValues484)
numSources=2
numLTerms=30
source1=SingleSource(0,0,1)
source2=SingleSource(1,0,2)
config=Configuration(numSources,pointSource484,np.array([source1,source2]))


class Optimization:
    def __init__(self,configuration,gtol=1e-5,temp=0.0001,numLTerms=30,maxStepSize=0.45):
        self.configuration=configuration
        self.localMinKwargs={"method":"L-BFGS-B", "jac":True, "options":{'gtol':gtol, 'disp':False}}
        self.gtol=gtol
        self.temp=temp
        self.numLTerms=numLTerms
        self.mybounds=customOptimizerSettings.MyBounds(self.configuration.numSources)
        self.mytakestep=customOptimizerSettings.MyTakeStep(self.configuration.numSources,stepsize=maxStepSize)
    
    #@njit
    def minimizerFunction(self,configVector):
        #return self.configuration.sigmaRMSSqu(self.numLTerms),self.configuration.sigmaRMSSquGrad(self.numLTerms)
        return sigmaRMSSquFuncs.sigmaRMSSqu(configVector,self.configuration.numSources,self.configuration.sourceType.aLVals,self.numLTerms), sigmaRMSSquFuncs.sigmaRMSSquGrad(configVector,self.configuration.numSources,self.configuration.sourceType.aLVals,self.numLTerms)
        
    def optimize(self, numIterations):
        result=basinhopping(self.minimizerFunction,self.configuration.configVector,minimizer_kwargs=self.localMinKwargs,niter=numIterations,T=self.temp,disp=True,accept_test=self.mybounds,take_step=self.mytakestep)
        self.configuration.configVector=np.array(list(result.x))
        return self.configuration.configVector

optimizer=Optimization(config)
print(optimizer.minimizerFunction(config.configVector))
optimizer.optimize(2)

print(optimizer.configuration.sourceTuples())
print(config.sourceTuples())




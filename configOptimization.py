import sigmaRMSSquFuncs
import customOptimizerSettings
from aLValueFuncs import *
import imperfectionFuncs
import rdofSigmaRMSSquFuncs
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
        
class Configuration: #rename to SourceConfiguration
    def __init__(self, numSources, sourceType, sourceArray=None, configVector=None, sourceTuples=None):
        self.numSources=numSources
        self.sourceType=sourceType
        
        if (sourceArray is None) and (configVector is None) and (sourceTuples is None):
            self.configVector=self.randomConfigVector()
            
        elif configVector is None and sourceTuples is None: #initialize from sourceArray
            if self.numSources==np.size(sourceArray):  #kept in as check (that got right num of sources in array!)
                self.configVector=self.configVectorFromSourceArray(sourceArray)
                
            else:
                #sourceArray
                raise ValueError('Number of sources did not match size of array')
                
        elif sourceArray is None and sourceTuples is None: #initialize from configVector
            if 3*self.numSources==np.size(configVector):  
                self.configVector=configVector
                
            else:
                #configVector
                raise ValueError('Number of sources did not match size of array')
                
        elif sourceArray is None and configVector is None: #initialize from sourceTuples
            if 3*self.numSources==np.size(sourceTuples):  #kept in as check (that got right num of sources in array!)
                self.configVector=self.configVectorFromSourceTuples(sourceTuples)
                
            else:
                #sourceTuples
                raise ValueError('Number of sources did not match size of array')
                
        else:
            raise ValueError('Arguments were not provided correctly.')
    
            
    def configVectorFromSourceArray(self,sourceArray):
        thetas=list(map(lambda x: x.theta, sourceArray))
        phis=list(map(lambda x: x.phi, sourceArray))
        strengths=list(map(lambda x: x.strength, sourceArray))
        return np.array([thetas,phis,strengths]).flatten()
    
    def configVectorFromSourceTuples(self, sourceTuples):
        return np.array(sourceTuples).T.flatten()
    
    def sourceArrayFromConfigVector(self):
        sourceArray=np.empty(self.numSources,dtype=SingleSource)
        for i in range(self.numSources):
            sourceArray[i]=SingleSource(self.configVector[i],self.configVector[i+self.numSources],self.configVector[i+2*self.numSources])
        return sourceArray
    
    def randomConfigVector(self):
        thetas=np.random.rand(self.numSources)*(np.pi)
        phis=np.random.rand(self.numSources)*(np.pi*2)
        strengths=np.ones(self.numSources) #all same strength
        return np.concatenate((thetas,phis,strengths))
    
    def sourceTuples(self):
        sourceArray=self.sourceArrayFromConfigVector()
        return list(map(lambda x: x.sourceTuple(),sourceArray))
    
    def sigmaRMSSqu(self,numLTerms):
        return sigmaRMSSquFuncs.sigmaRMSSqu(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms)

    def sigmaRMSSquGrad(self,numLTerms):
        return sigmaRMSSquFuncs.sigmaRMSSquGrad(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms)

    def sigmaRMSPercent(self,numLTerms):
        return 100*np.sqrt(self.sigmaRMSSqu(numLTerms))

    def stabilityTest(self,numLTerms,numPerturbations,stdTheta,stdPhi,stdStr):
        return imperfectionFuncs.stabilityTest(self.configVector,self.numSources,self.sourceType.aLVals,numLTerms,numPerturbations,stdTheta,stdPhi,stdStr)
    
    def minimizerFunction(self, configVector, numLTerms): #must take in the config vector separately so that it works with the basinhopping optimization function
        return sigmaRMSSquFuncs.sigmaRMSSqu(configVector,self.numSources,self.sourceType.aLVals,numLTerms), sigmaRMSSquFuncs.sigmaRMSSquGrad(configVector,self.numSources,self.sourceType.aLVals,numLTerms)


class ConfigurationRDOF(Configuration): #rename      
    def __init__(self, numSources, sourceType, sourceArray=None, configVector=None, sourceTuples=None):
        self.numSources=numSources
        self.sourceType=sourceType
        
        if (sourceArray is None) and (configVector is None) and (sourceTuples is None):
            self.configVector=self.randomConfigVector()
            
        elif configVector is None and sourceTuples is None: #initialize from sourceArray
            if int(numSources/2)==np.size(sourceArray):  #kept in as check (that got right num of sources in array!)
                self.configVector=self.configVectorFromSourceArray(sourceArray)
                
            else:
                #sourceArray
                raise ValueError('Number of sources was not twice size of array')
                
        elif sourceArray is None and sourceTuples is None: #initialize from configVector
            if 3*int(numSources/2)==np.size(configVector):  
                self.configVector=configVector
                
            else:
                #configVector
                raise ValueError('Number of sources was not twice size of array')
                
        elif sourceArray is None and configVector is None: #initialize from sourceTuples
            if 3*int(numSources/2)==np.size(sourceTuples):  #kept in as check (that got right num of sources in array!)
                self.configVector=self.configVectorFromSourceTuples(sourceTuples)
                
            else:
                #sourceTuples
                raise ValueError('Number of sources was not twice size of array')
                
        else:
            raise ValueError('Arguments were not provided correctly.')

    def fullConfigVector(self):
        fullConfigVec=np.zeros(3*self.numSources)
        halfNumSources=int(self.numSources/2)
        for i in range(halfNumSources):
            fullConfigVec[i]=self.configVector[i]
            fullConfigVec[i+halfNumSources]=np.pi-self.configVector[i]
            fullConfigVec[i+2*halfNumSources]=self.configVector[i+halfNumSources]
            fullConfigVec[i+3*halfNumSources]=np.pi+self.configVector[i+halfNumSources]
            fullConfigVec[i+4*halfNumSources]=self.configVector[i+2*halfNumSources]
            fullConfigVec[i+5*halfNumSources]=self.configVector[i+2*halfNumSources]
        return fullConfigVec
    
    def randomConfigVector(self):
        thetas=np.random.rand(int(self.numSources/2))*(np.pi)
        phis=np.random.rand(int(self.numSources/2))*(np.pi*2)
        strengths=np.ones(int(self.numSources/2)) #all same strength
        return np.concatenate((thetas,phis,strengths))
    
    def fullSourceArray(self):
        fullConfigVec=self.fullConfigVector()
        sourceArray=np.empty(self.numSources,dtype=SingleSource)
        for i in range(self.numSources):
            sourceArray[i]=SingleSource(fullConfigVec[i],fullConfigVec[i+self.numSources],fullConfigVec[i+2*self.numSources])
        return sourceArray
    
    def sourceArrayFromConfigVector(self):
        sourceArray=np.empty(int(self.numSources/2),dtype=SingleSource)
        for i in range(int(self.numSources/2)):
            sourceArray[i]=SingleSource(self.configVector[i],self.configVector[i+int(self.numSources/2)],self.configVector[i+2*int(self.numSources/2)])
        return sourceArray
    
    def fullSourceTuples(self):
        sourceArray=self.fullSourceArray()
        return list(map(lambda x: x.sourceTuple(),sourceArray))
    
    def sigmaRMSSqu(self,numLTerms):
        return rdofSigmaRMSSquFuncs.sigmaRMSSquRDOF(self.configVector,int(self.numSources/2),self.sourceType.aLVals,numLTerms)
        
    def sigmaRMSSquGrad(self,numLTerms):
        return rdofSigmaRMSSquFuncs.sigmaRMSSquGradRDOF(self.configVector,int(self.numSources/2),self.sourceType.aLVals,numLTerms)

    def minimizerFunction(self, configVector, numLTerms):
        return rdofSigmaRMSSquFuncs.sigmaRMSSquRDOF(configVector,int(self.numSources/2),self.sourceType.aLVals,numLTerms),rdofSigmaRMSSquFuncs.sigmaRMSSquGradRDOF(configVector,int(self.numSources/2),self.sourceType.aLVals,numLTerms)


class Optimization:
    def __init__(self,configuration,gtol=1e-5,temp=0.0001,numLTerms=30,maxStepSize=0.45):
        self.configuration=configuration
        self.gtol=gtol
        self.temp=temp
        self.numLTerms=numLTerms
        if isinstance(self.configuration, ConfigurationRDOF)==False:
            self.mybounds=customOptimizerSettings.MyBounds(self.configuration.numSources)
            self.mytakestep=customOptimizerSettings.MyTakeStep(self.configuration.numSources,stepsize=maxStepSize)
        else:
            self.mytakestep=customOptimizerSettings.MyTakeStepRDOF(int(self.configuration.numSources/2),stepsize=maxStepSize)
            self.mybounds=customOptimizerSettings.MyBounds(int(self.configuration.numSources/2))
        self.localMinKwargs={"method":"L-BFGS-B", "jac":True, "args":(self.numLTerms),"options":{'gtol':self.gtol, 'disp':False}}

    
    def optimize(self, numIterations):
        result=basinhopping(self.configuration.minimizerFunction,self.configuration.configVector,minimizer_kwargs=self.localMinKwargs,niter=numIterations,T=self.temp,disp=True,accept_test=self.mybounds,take_step=self.mytakestep)
        self.configuration.configVector=np.array(list(result.x))
        return self.configuration.configVector
    










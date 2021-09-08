import numpy as np

class MyTakeStep:
    def __init__(self, numSources,stepsize=0.45):
        self.stepsize=stepsize
        self.numSources=numSources
        self.rng=np.random.default_rng()
    def __call__(self, x):
        s=self.stepsize
        x[0:2*self.numSources]+=self.rng.uniform(-s,s,x[0:2*self.numSources].shape)
        #x[2*self.numSources:]=x[2*self.numSources:]*self.rng.uniform(0,2,x[2*self.numSources:].shape) #multiply by random number between 0 and 2 - same as max step size of itself
        x[2*self.numSources:]+=self.rng.uniform(-1,1,x[2*self.numSources:].shape)
        return x

class MyBounds:
    def __init__(self, numSources):
        self.xmin = np.array([0 for i in range(numSources)])
        self.numSources=numSources
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x[2*self.numSources:] >= self.xmin))
        return tmin

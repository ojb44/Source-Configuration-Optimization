from scipy.integrate import quad
import numpy as np
from scipy.special import eval_legendre

#Point sources

def pointaLIntegrand(x,l,d,a):
    return (d*x - a)/((d**2 + a**2 - 2*a*d*x)**(3/2))*eval_legendre(l,x)

def aL(l,d,a):
    aLNormFactor=0.5*quad(pointaLIntegrand, a/d,1,args=(0,d,a))[0]
    I=quad(pointaLIntegrand, a/d,1,args=(l,d,a))
    return ((2*l+1)/2)*I[0]/aLNormFactor

aLValues484=np.array([aL(l,4.84,1) for l in range(50)])

aLValues2587=np.array([aL(l,2.58733,1) for l in range(50)])

#Laser light

def laseraLIntegrand(x,l,etaPerp,beta,alpha,I0):
    return I0*(1-(1-etaPerp)**(x**3))*np.exp(-(((1-x**2)/(alpha**2))**(beta/2)))*x*eval_legendre(l,x)

def laseraL(l,etaPerp,beta,alpha,I0):
    aLNormFactor=0.5*quad(laseraLIntegrand,0,1,args=(0,etaPerp,beta,alpha,I0))[0]
    I=quad(laseraLIntegrand,0,1,args=(l,etaPerp,beta,alpha,I0))
    return ((2*l+1)/2)*I[0]/aLNormFactor

aLValuesLaser=np.array([laseraL(l,0.9,5,0.95,1) for l in range(50)])
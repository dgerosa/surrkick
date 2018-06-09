'''surrkick: Black-hole kicks from numerical-relativity surrogate models.

surrkick is a python module to extract radiated energy and momenta from waveform approximants.
The present version of the code uses the numerical-relativity surrogate model NRSur7dq2.

More info on the code available in our paper (Gerosa, Hebert, Stein 2018) and at https://davidegerosa.com/surrkick/
Surrkick is distributed through the Python Package index (https://pypi.python.org/pypi/surrkick)
and GitHub (https://github.com/dgerosa/surrkick).
'''

from __future__ import print_function,division
import sys
import os
import time
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import NRSur7dq2
from singleton_decorator import singleton
import h5py
from tqdm import tqdm
import cPickle as pickle
import multiprocessing, pathos.multiprocessing
import precession
import warnings
import json


__author__ = "Davide Gerosa, Francois Hebert, Leo Stein"
__email__ = "dgerosa@caltech.edu"
__license__ = "MIT"
__version__ = "1.1.1"
__doc__+="\n\n"+"Authors: "+__author__+"\n"+\
        "email: "+__email__+"\n"+\
        "Licence: "+__license__+"\n"+\
        "Version: "+__version__



class summodes(object):
    '''Return indexes to perform a mode sum in l (from 0 to lmax) and m (from -l to +l).'''

    @staticmethod
    def single(lmax):
        '''Single mode sum: Sum_{l,m} with 0<=l<=lmax and -l<=m<=l. Returns a list of (l,m) tuples.
        Usage: iterations=surrkick.summodes.single(lmax)'''

        iterations=[]
        for l1 in np.arange(2,lmax+1):
            for m1 in np.arange(-l1,l1+1):
                    iterations.append((l1,m1))
        return iterations

    @staticmethod
    def double(lmax):
        '''Double mode sum: Sum_{l1,m1} Sum_{l2,m2}  with 0<=l1,l2<=lmax and -l<=m<=l. Returns a list of (l1,m1,l2,m2) tuples.
        Usage: iterations=surrkick.summodes.double(lmax)'''

        iterations=[]
        for l1 in np.arange(2,lmax+1):
            for m1 in np.arange(-l1,l1+1):
                for l2 in np.arange(2,lmax+1):
                    for m2 in np.arange(-l2,l2+1):
                        iterations.append((l1,m1,l2,m2))

        return iterations


class convert(object):
    '''Utility class to convert units to other units.'''

    @staticmethod
    def kms(x):
        '''Convert a velocity from natural units (c=1) to km/s.
        Usage: vkms=surrkick.convert.kms(vnat)'''

        return x * 299792.458

    @staticmethod
    def cisone(x):
        '''Convert a velocity from km/s to natural units (c=1)
        Usage: vnat=surrkick.convert.cisone(vkms)'''

        return x / 299792.458


@singleton
class surrogate(object):
    '''Initialize the NRSur7dq2 surrogate model described in arXiv:1705.07089. The only purpose of this class is to wrap the surrogate with a singleton pattern (which means there can only be one instance of this class, and creation of additional instances just return the first one).'''

    def __init__(self):
        '''Placeholder'''
        self._sur=None

    def sur(self):
        '''Load surrogate from file NRSur7dq2.h5.
        Usage: sur=surrkick.surrogate().sur()'''

        if self._sur==None:
            self._sur= NRSur7dq2.NRSurrogate7dq2()
            #self._sur = NRSur7dq2.NRSurrogate7dq2('NRSur7dq2.h5')
        return self._sur


class surrkick(object):
    '''Extract energy, linear momentum and angular momentum emitted in gravitational waves from a waveform surrogate model. We use a frame where the orbital angular momentum is along the z axis and the heavier (lighter) is on the positive (negative) x-axis at the reference time `t_ref`.
    Usage: sk=surrkick.surrkick(q=1,chi1=[0,0,0],chi2=[0,0,0],t_ref=-100)
    Parameters:
    - `q`: binary mass ratio in the range 1:1 to 2:1. Can handle both conventions: q in [0.5,1] or [1,2].
    - `chi1`: spin vector of the heavier BH.
    - `chi2`: spin vector of the lighter BH.
    - `t_ref`: reference time at which spins are specified (must be -4500<=t_ref<=-100)
    '''

    def __init__(self,q=1,chi1=[0,0,0],chi2=[0,0,0],t_ref=-100):
        '''Initialize the `surrkick` class.'''

        self.sur=surrogate().sur()
        '''Initialize the surrogate. Note it's a singleton'''

        self.q = max(q,1/q)
        '''Binary mass ratio in the range 1:1 to 2:1.
        Usage: q=surrkick.surrkick().q'''

        self.chi1 = np.array(chi1)
        '''Spin vector of the heavier BH.
        Usage: chi1=surrkick.surrkick().chi1'''

        self.chi2 = np.array(chi2)
        '''Spin vector of the lighter BH.
        Usage: chi2=surrkick.surrkick().chi2'''

        self.times = self.sur.t_coorb
        '''Time nodes where quantities are evaluated.
        Usage: times=surrkick.surrkick().times'''

        # Check the reference time makes sense
        self.t_ref=t_ref
        '''Reference time at which spins are specified (must be -4500<=t_ref<=-100).
        Usage: t_ref=surrkick.surrkick().t_ref'''
        assert self.t_ref>=-4500 and self.t_ref<=-100 # Check you're in the regions where spins are OK.

        # Hidden variables for lazy loading
        self._hsample = None
        self._hdotsample = None
        self._lmax = None
        self._dEdt = None
        self._Eoft = None
        self._Erad = None
        self._Moft = None
        self._Mrad = None
        self._Mfin = None
        self._dPdt = None
        self._Poft = None
        self._Prad = None
        self._voft = None
        self._kick = None
        self._dJdt = None
        self._Joft = None
        self._Jrad = None
        self._xoft = None

    class coeffs(object):
        '''Coefficients of the momentum expression, from Eqs. (3.16-3.19,3.25) of arXiv:0707.4654. All are defined as static methods and can be called with, e.g., surrkick.surrkick().self.coeffs.a(l,m).'''

        @staticmethod
        def a(l,m):
            '''Eq. (3.16) of arXiv:0707.4654.
            Usage: a=surrkick.surrkick().coeffs.a(l,m)'''

            return ( (l-m) * (l+m+1) )**0.5 / ( l * (l+1) )

        @staticmethod
        def b(l,m):
            '''Eq. (3.17) of arXiv:0707.4654.
            Usage: b=surrkick.surrkick().coeffs.b(l,m)'''

            return  ( 1/(2*l) ) *  ( ( (l-2) * (l+2) * (l+m) * (l+m-1) ) / ( (2*l-1) * (2*l+1) ))**0.5

        @staticmethod
        def c(l,m):
            '''Eq. (3.18) of arXiv:0707.4654.
            Usage: c=surrkick.surrkick().coeffs.c(l,m)'''

            return  2*m / ( l * (l+1) )

        @staticmethod
        def d(l,m):
            '''Eq. (3.19) of arXiv:0707.4654.
            Usage: d=surrkick.surrkick().coeffs.d(l,m)'''

            return  ( 1/l ) *  ( ( (l-2) * (l+2) * (l-m) * (l+m) ) / ( (2*l-1) * (2*l+1) ))**0.5

        @staticmethod
        def f(l,m):
            '''Eq. (3.25) of arXiv:0707.4654.
            Usage: `f=surrkick.coeffs.f(l,m)`'''

            return  ( l*(l+1) - m*(m+1) )**0.5

    @property
    def hsample(self):
        '''Modes of the gravitational-wave strain h=hp-i*hc evaluated at the surrogate time nodes. Returns a dictiornary with keys (l,m).
        Usage: hsample=surrkick.surrkick().hsample; hsample[l,m]'''

        if self._hsample is None:

            if self.t_ref==-4500: # This is the default value for NRSur7dq2, which wants a None
                self._hsample = self.sur(self.q, self.chi1, self.chi2,t=self.times,t_ref=None)
            else:
                self._hsample = self.sur(self.q, self.chi1, self.chi2,t=self.times,t_ref=self.t_ref)

        # Returns a python dictionary with keys (l,m)
        return self._hsample

    @property
    def lmax(self):
        '''Largest l mode available. lmax=4 for NRSur7dq2.
        Usage: lmax=surrkick.surrkick().lmax'''

        if self._lmax is None:
            self._lmax = sorted(self.hsample.keys())[-1][0]
            #self._lmax=3 # To cut all extractions at l=3
        return self._lmax

    def h(self,l,m):
        '''Correct the strain values to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used). Returns a single mode.
        Usage: hlm=surrkick.surrkick().h(l,m)'''

        if l<2 or l>self.lmax:
            return np.zeros(len(self.times),dtype=complex)
        elif m<-l or m>l:
            return np.zeros(len(self.times),dtype=complex)
        else:
            return self.hsample[l,m]

    @property
    def hdotsample(self):
        '''Derivative of the strain at the time nodes. First interpolate the h with a standard spline, then derivate the spline and evaluate that derivative at the time nodes. Returns a dictiornary with keys (l,m).
        Usage: hdotsample=surrkick.surrkick().hdotsample; hdotsample[l,m]'''

        if self._hdotsample is None:

            # Derivatives with splines
            self._hdotsample =  {k: spline(self.times,v.real).derivative()(self.times)+1j*spline(self.times,v.imag).derivative()(self.times) for k, v in self.hsample.items()}

            # Derivatives with finite differencing
            #self._hdotsample = {k: np.gradient(v,edge_order=2)/np.gradient(self.times,edge_order=2) for k, v in self.hsample.items()}

        return self._hdotsample

    def hdot(self,l,m):
        '''Correct the hdot values to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used). Returns a single mode.
        Usage: hdotlm=surrkick.surrkick().hdot(l,m)'''

        if l<2 or l>self.lmax:
            return np.zeros(len(self.times),dtype=complex)
        elif m<-l or m>l:
            return np.zeros(len(self.times),dtype=complex)
        else:
            return self.hdotsample[l,m]

    @property
    def dEdt(self):
        '''Implement Eq. (3.8) of arXiv:0707.4654 for the energy momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.
        Usage: dEdt=surrkick.surrkick().dEdt
        '''

        if self._dEdt is None:
            dEdt = 0
            for l,m in summodes.single(self.lmax):

                # Eq. (3.8)
                dEdt += (1/(16*np.pi)) * np.abs(self.hdot(l,m))**2

            self._dEdt = dEdt

        return self._dEdt

    @property
    def Eoft(self):
        '''Radiated energy as a function of time. Time integral of Eq. (3.8) of arXiv:0707.4654, evaluated at the time nodes. We first interpolate with a spline and then integrate analytically.
        Usage: Eoft=surrkick.surrkick().Eoft'''

        if self._Eoft is None:
            # The integral of a spline is called antiderivative (mmmh...)
            origin=0
            self._Eoft = spline(self.times,self.dEdt).antiderivative()(self.times)-origin

            flag_2PN=False

            if not flag_2PN:
                #Estimate energy radiated before the start of the simulation using 0PN result (Peters 64)
                tbuffer=100
                tend=self.times[0]+tbuffer
                Edot0 = spline(self.times[self.times<tend],self.dEdt[self.times<tend]).antiderivative()(tend)/tbuffer
                E0 = ( (5./1024.) *(self.q/(1.+self.q)**2.)**3. * Edot0 )**(1./5.)
                self._Eoft+=E0

            else:
                # Estimate the energy emitted before the start of the surrogate waveform using a 2PN expression.
                # Extract the orbital phase evolution from the surrogate
                if self.t_ref==-4500: # This is the default value for NRSur7dq2, which wants a None
                    dummy,phi,dummy,dummy=self.sur.get_dynamics(self.q,self.chi1,self.chi2,t_ref=None)
                else:
                    dummy,phi,dummy,dummy=self.sur.get_dynamics(self.q,self.chi1,self.chi2,t_ref=self.t_ref)
                # Interpolate and derivate to find the orbital frequency as t_ref
                omega = spline(self.sur.tds, phi).derivative()(self.t_ref)
                # 2PN expression for the binding energy. Equations from Arun+ gr-qc/0810.5336v3 (note v3 because of an erratum!)
                m1=self.q/(1+self.q)             # m1 is the heavy BH and q>1
                m2=1/(1+self.q)                  # m2 is the light BH and q>1
                nu = m1*m2                       # Eq. 2.12 with M=1
                delta = m1-m2                    # Eq. 2.14 with M=1
                chis = (self.chi1 + self.chi2)/2 # Eq. 2.17
                chia = (self.chi1 - self.chi2)/2 # Eq. 2.18
                Lhat = np.array([0,0,1])         # L at t_ref
                # Eqs. C1-C4
                Enewt = -nu/2
                E2 = -3/4 -nu/12
                E3 = (8/3 -4*nu/3) * np.dot(chis,Lhat) + (8/3)*delta*np.dot(chia,Lhat)
                E4 = -27/8 +19*nu/8 -(nu**2)/24 +nu*( np.dot(chis,chis)-np.dot(chia,chia) - 3* ( np.dot(chis,Lhat)**2 - np.dot(chia,Lhat)**2 )  ) \
                      + (1/2 - nu) * ( np.dot(chis,chis) + np.dot(chia,chia) - 3* ( np.dot(chis,Lhat)**2 + np.dot(chia,Lhat)**2 )  ) \
                      + delta * ( np.dot(chis,chia) - 3*np.dot(chis,Lhat)*np.dot(chia,Lhat))
                v = omega**(1/3) # Usual PN velocity/frequency definition
                Ebinding = (Enewt*v**2) * (1+ E2*v**2 + E3*v**3 + E4*v**4) # Eq. 6.18

                self._Eoft-=Ebinding



        return self._Eoft

    @property
    def Erad(self):
        ''' Total energy radiated, i.e. E(t) at the last time node.
        Usage: Erad=surrkick.surrkick().Erad'''

        if self._Erad is None:
            self._Erad = self.Eoft[-1]
        return self._Erad

    @property
    def Moft(self):
        ''' Mass profile in units of the mass at the beginning of the surrogate.
        Usage: Moft=surrkick.surrkick().Moft'''


        if self._Moft is None:
            self._Moft = 1-self.Eoft+self.Eoft[0]

        return self._Moft

    @property
    def Mrad(self):
        ''' Final mass in units of the mass at the beginning of the surrogate.
        Usage: Mrad=surrkick.surrkick().Mrad'''

        if self._Mrad is None:
            self._Mrad = self.Moft[-1]

        return self._Mrad

    @property
    def Mfin(self):
        ''' Final mass in units of the mass at the early times (t=-infinity).
        Usage: Mfin=surrkick.surrkick().Mfin'''

        if self._Mfin is None:
            self._Mfin = 1 - self.Eoft[-1]/(1+self.Eoft[0])

        return self._Mfin

    @property
    def dPdt(self):
        '''Implement Eq. (3.14-3.15) of arXiv:0707.4654 for the three component of the linear momentum momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there. Returned array has size len(times)x3.
        Usage: dPdt=surrkick.surrkick().dPdt'''

        if self._dPdt is None:

            dPpdt = 0
            dPzdt = 0

            for l,m in summodes.single(self.lmax):

                # Eq. 3.14. dPpdt= dPxdt + i dPydt
                dPpdt += (1/(8*np.pi)) * self.hdot(l,m) * ( self.coeffs.a(l,m) * np.conj(self.hdot(l,m+1)) + self.coeffs.b(l,-m) * np.conj(self.hdot(l-1,m+1)) - self.coeffs.b(l+1,m+1) * np.conj(self.hdot(l+1,m+1)) )
                # Eq. 3.15
                dPzdt += (1/(16*np.pi)) * self.hdot(l,m) * ( self.coeffs.c(l,m) * np.conj(self.hdot(l,m)) + self.coeffs.d(l,m) * np.conj(self.hdot(l-1,m)) + self.coeffs.d(l+1,m) * np.conj(self.hdot(l+1,m)) )

            dPxdt=dPpdt.real # From the definition of Pplus
            dPydt=dPpdt.imag # From the definition of Pplus

            assert max(dPzdt.imag)<1e-6 # Check...
            dPzdt=dPzdt.real # Kill the imaginary part

            self._dPdt = np.transpose([dPxdt,dPydt,dPzdt])

        return self._dPdt

    @property
    def Poft(self):
        '''Radiated linear momentum as a function of time. Time integral of Eq. (3.14-3.15) of arXiv:0707.4654, evaluated at the time nodes. We first interpolate with a spline and then integrate analytically.
        Usage: Poft=surrkick.surrkick().Poft'''

        if self._Poft is None:
            # The integral of a spline is called antiderivative (mmmh...)
            origin=[0,0,0]
            self._Poft = np.transpose([spline(self.times,v).antiderivative()(self.times)-o  for v,o in zip(np.transpose(self.dPdt),origin)])

            # Eliminate unphysical drift due to the starting point of the integration. Integrate for tbuffer and substract the mean.
            tbuffer=1000
            tend=self.times[0]+tbuffer
            P0 = np.array([spline(self.times[self.times<tend],v[self.times<tend]).antiderivative()(tend)/tbuffer  for v in np.transpose(self._Poft)])
            self._Poft-=P0

        return self._Poft

    @property
    def Prad(self):
        '''Total linear momentum radiated, i.e. the norm of [Px(t),Py(t),Pz(t)] at the last time node.
        Usage: Prad=surrkick.surrkick().Prad'''

        if self._Prad is None:
            self._Prad = np.linalg.norm(self.Poft[-1])

        return self._Prad

    @property
    def voft(self):
        '''Velocity of the center of mass, i.e. minus the momentum radiated over mass.
        Usage: voft=surrkick.surrkick().voft'''

        if self._voft is None:
            self._voft= np.array([-x/y for x,y in zip(self.Poft,self.Moft)])

        return self._voft

    @property
    def kick(self):
        '''Final kick velocity.
        Usage: kick=surrkick.surrkick().kick'''

        if self._kick is None:
            self._kick = np.linalg.norm(self.voft[-1])

        return self._kick

    @property
    def kickcomp(self):
        '''Components of the final kick in the xyz frame of the surrogate
        Usage: vkx,vky,vkz=surrkick.surrkick().kick'''

        return self.voft[-1]

    @property
    def kickdir(self):
        '''Direction of the final kick in the xyz frame of the surrogate
        Usage: vkhatx,vkhaty,vkhatz=surrkick.surrkick().kick'''

        return self.kickcomp/self.kick


    @property
    def dJdt(self):
        '''Implement Eq. (3.22-3.24) of arXiv:0707.4654 for the three component of the angular momentum momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there. Returned array has size len(times)x3.
        Usage: dPdt=surrkick.surrkick().dPdt'''

        if self._dJdt is None:

            dJxdt = 0
            dJydt = 0
            dJzdt = 0

            for l,m in summodes.single(self.lmax):

                # Eq. 3.22
                dJxdt += (1/(32*np.pi)) * self.h(l,m) * ( self.coeffs.f(l,m) * np.conj(self.hdot(l,m+1)) + self.coeffs.f(l,-m) * np.conj(self.hdot(l,m-1)) )
                # Eq. 3.23
                dJydt += (-1/(32*np.pi)) * self.h(l,m) * ( self.coeffs.f(l,m) * np.conj(self.hdot(l,m+1)) - self.coeffs.f(l,-m) * np.conj(self.hdot(l,m-1)) )
                # Eq. 3.24
                dJzdt += (1/(16*np.pi)) * m * self.h(l,m) * np.conj(self.hdot(l,m))

            dJxdt=dJxdt.imag
            dJydt=dJydt.real
            dJzdt=dJzdt.imag

            self._dJdt = np.transpose([dJxdt,dJydt,dJzdt])

        return self._dJdt

    @property
    def Joft(self):
        '''Radiated angular momentum as a function of time. Time integral of Eq. (3.22-3.24) of arXiv:0707.4654, evaluated at the time nodes. We first interpolate with a spline and then integrate analytically.
        Usage: Joft=surrkick.surrkick().Joft'''

        if self._Joft is None:

            # The integral of a spline is called antiderivative (mmmh...)
            origin=[0,0,0]
            self._Joft = np.transpose([spline(self.times,v).antiderivative()(self.times)-o  for v,o in zip(np.transpose(self.dJdt),origin)])

        return self._Joft

    @property
    def Jrad(self):
        '''Total angular momentum radiated, i.e. the norm of [Jx(t),Jy(t),Jz(t)] at the last time node.
        Usage: Jrad=surrkick.surrkick().Jrad'''

        if self._Jrad is None:
            self._Jrad = np.linalg.norm(self.Joft[-1])
        return self._Jrad

    @property
    def xoft(self):
        '''Trajectory of the spacetime center of mass, i.e. time integral of -P(t), where P is the linear momentum radiated in GWs.
        Usage: xoft=surrkick.surrkick().xoft'''

        if self._xoft is None:
            # The integral of a spline is called antiderivative (mmmh...)
            origin=[0,0,0]
            self._xoft = np.transpose([spline(self.times,v).antiderivative()(self.times)-vo*self.times  for v,vo in zip(np.transpose(self.voft),origin)])
        return self._xoft


def project(timeseries,direction):
    '''Project a 3D time series along some direction.
    Usage projection=project(timeseries, direction)'''

    return np.array([np.dot(t,direction) for t in timeseries])


@np.vectorize
def alphakick(q,chi1mag,chi2mag,theta1,theta2,deltaphi,alpha,t_ref=-100):
    '''Return kick as a function of relative orientation of the spins and an overall rotation alpha.
    Usage: kick=alphakick(q,chi1mag,chi2mag,theta1,theta2,deltaphi,alpha)'''

    chi1 = [chi1mag*np.sin(theta1)*np.cos(alpha),chi1mag*np.sin(theta1)*np.sin(alpha),chi1mag*np.cos(theta1)]
    chi2 = [chi2mag*np.sin(theta2)*np.cos(deltaphi+alpha),chi2mag*np.sin(theta2)*np.sin(deltaphi+alpha),chi1mag*np.cos(theta2)]
    return surrkick(q=q,chi1=chi1,chi2=chi2,t_ref=t_ref).kick

def maxalpha(q=1,chi1mag=0,chi2mag=0.,theta1=0.,theta2=0.,deltaphi=0.,t_ref=-100,nguess=20):
    '''For a given relative configration (theta1, theta2, deltaphi), find the min and max of the kick varying over alpha.
    Usage: kickmin,kickmax,alphamin,alphamax=maxalpha(q=1,chi1mag=0,chi2mag=0.,theta1=0.,theta2=0.,deltaphi=0.,t_ref=-100)'''

    # Generate some guesses
    alpha_vals=np.linspace(-np.pi,np.pi,int(nguess*2+1))[1:-1:2]
    guess=alphakick(q,chi1mag,chi2mag,theta1,theta2,deltaphi,alpha_vals)
    # Find suitable bounds in alpha given the max/min array position
    def _boundsfromid(id):
        if id==0: # Guess is at the beginning
            return (-np.pi,alpha_vals[id+1])
        elif id==len(alpha_vals)-1: # Guess is at the end
            return (alpha_vals[id-1],np.pi)
        else: # Guess is in the middle
            return (alpha_vals[id-1],alpha_vals[id+1])

    # Find minimum
    resmin = scipy.optimize.minimize_scalar(lambda alpha: alphakick(q,chi1mag,chi2mag,theta1,theta2,deltaphi,alpha), bounds=_boundsfromid(np.argmin(guess)),  method='bounded')

    # Find maximum
    resmax = scipy.optimize.minimize_scalar(lambda alpha: -alphakick(q,chi1mag,chi2mag,theta1,theta2,deltaphi,alpha), bounds=_boundsfromid(np.argmax(guess)),  method='bounded')

    return resmin.fun, -resmax.fun, resmin.x, resmax.x




class plots(object):
    '''Reproduce plots of our paper: Black-hole kicks from numerical-relativity surrogate models'''

    def plottingstuff(function):
        '''Python decorator to handle plotting, including defining all defaults and storing the final pdf. Just add @plottingstuff to any methond of the plots class.'''

        def wrapper(self):
            print("Plotting:", function.__name__+".pdf")

            # Before function call
            global plt,AutoMinorLocator,MultipleLocator,LogLocator,NullFormatter
            from matplotlib import use #Useful when working on SSH
            use('Agg')
            from matplotlib import rc
            font = {'family':'serif','serif':['cmr10'],'weight' : 'medium','size' : 16}
            rc('font', **font)
            rc('text',usetex=True)
            #rc.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
            #rc('text.latex',preamble=r"\usepackage{amsmath}")
            import matplotlib
            matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
            rc('figure',max_open_warning=1000)
            rc('xtick',top=True)
            rc('ytick',right=True)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.ticker import AutoMinorLocator,MultipleLocator,LogLocator,NullFormatter
            pp= PdfPages(function.__name__+".pdf")

            fig = function(self)
            # Handle multiple figures
            try:
                len(fig)
            except:

                fig=[fig]
            for f in tqdm(fig):

                # Filter our annoying "elementwise comparison failed" warning (something related to the matplotlib backend and future versions)
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)

                    f.savefig(pp, format='pdf',bbox_inches='tight')
                f.clf()
            pp.close()
        return wrapper

    def animate(function):
        '''Python decorator to handle animations, including defining all defaults and storing the final movie. Just add @animate to any methond of the plots class.'''

        #def wrapper(*args, **kw):
        def wrapper(self):
            print("Animation:", function.__name__+".mp4")

            # Before function call
            global plt,AutoMinorLocator,MultipleLocator
            from matplotlib import use #Useful when working on SSH
            use('Agg')
            from matplotlib import rc
            font = {'family':'serif','serif':['cmr10'],'weight' : 'medium','size' : 16}
            rc('font', **font)
            rc('text',usetex=True)
            import matplotlib
            matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
            matplotlib.rcParams['figure.max_open_warning']=int(1e4)
            rc('figure',max_open_warning=100000)
            rc('xtick',top=True)
            rc('ytick',right=True)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.ticker import AutoMinorLocator,MultipleLocator

            encodeonly=False # if you already have the frames and want to test encoding options
            if not encodeonly:
                figs = function(self)
                try:
                    len(figs[0])
                except:
                    figs=[figs]
            else:
                figs=[[1,2,3],[1,2,3],[1,2,3]]

            for j,fig in enumerate(tqdm(figs)):

                if fig != [None]:

                    for i,f in enumerate(tqdm(fig)):

                        with warnings.catch_warnings():
                            warnings.simplefilter(action='ignore', category=FutureWarning)
                            framename = function.__name__+"_"+str(j)+"_"+"%05d.png"%i
                            if not encodeonly:
                                f.savefig(framename, bbox_inches='tight',format='png',dpi = 300)
                                f.clf()
                                plt.close(f)

                    rate = 100 #The movie is faster if this number is large
                    command ='ffmpeg -r '+str(rate)+' -i '+function.__name__+'_'+str(j)+'_'+'%05d.png -vcodec libx264 -crf 18 -y -an '+function.__name__+'_'+str(j)+'.mp4 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
                    print(command)
                    os.system(command)

                    if False:
                        command="rm -f "+function.__name__+"_"+str(j)+"*.png temp"
                        os.system(command)

        return wrapper

    @classmethod
    def minimal(self):
        '''Minimal working example.
        Usage: surrkick.plots.minimal()'''

        import matplotlib.pyplot as plt
        sk=surrkick(q=0.5,chi1=[0.8,0,0], chi2=[-0.8,0,0])
        print("vk/c=", sk.kick)
        plt.plot(sk.times,sk.voft[:,0],label="x")
        plt.plot(sk.times,sk.voft[:,1],label="y")
        plt.plot(sk.times,sk.voft[:,2],label="z")
        plt.plot(sk.times,project(sk.voft,sk.kickdir),label="vk")
        plt.xlim(-100,100)
        plt.legend()
        plt.show()

    @classmethod
    @plottingstuff
    def nospinprofiles(self):
        '''Fig. 1. Kick profiles for non-spinning binaries.
        Usage: surrkick.plots.nospinprofiles()'''

        L=0.7
        H=0.3
        S=0.05
        figP = plt.figure(figsize=(6,6))
        axP = [figP.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
        figE = plt.figure(figsize=(6,6))
        axE = figE.add_axes([0,0,L,H])
        figJ = plt.figure(figsize=(6,6))
        axJ = [figJ.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
        axPt = [ax.twinx() for ax in axP]

        q_vals = np.linspace(1,0.5,8)
        for i,q in enumerate(tqdm(q_vals)):
            color=plt.cm.copper(i/len(q_vals))
            b = surrkick(q=q)
            dashes=''
            axP[0].plot(b.times,b.voft[:,0]*1000,color=color,alpha=0.9,dashes=dashes)
            axP[1].plot(b.times,b.voft[:,1]*1000,color=color,alpha=0.9,dashes=dashes)
            axP[2].plot(b.times,b.voft[:,2]*1000,color=color,alpha=0.9,dashes=dashes)
            axP[3].plot(b.times,project(b.voft,b.kickdir)*1000,color=color,alpha=0.9,dashes=dashes)
            axE.plot(b.times,b.Eoft,color=color,alpha=0.7)
            axJ[0].plot(b.times,b.Joft[:,0],color=color,alpha=0.7,dashes=dashes)
            axJ[1].plot(b.times,b.Joft[:,1],color=color,alpha=0.7,dashes=dashes)
            axJ[2].plot(b.times,b.Joft[:,2],color=color,alpha=0.7,dashes=dashes)
            axJ[3].plot(b.times,project(b.Joft,b.Joft[-1]),color=color,alpha=0.7,dashes=dashes)

        for ax in axP+[axE]+axJ:
            ax.set_xlim(-50,50)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        for ax in axP[:-1]+axJ[:-1]:
            ax.set_xticklabels([])
        for ax in [axP[-1]]+[axJ[-1]]+[axE]:
            ax.set_xlabel("$t\;\;[M]$")
        for ax,axt,d in zip(axP,axPt,["x","y","z","v_k"]):
            ax.set_ylim(-1,1)
            ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")
            axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/100)
            axt.xaxis.set_minor_locator(AutoMinorLocator())
            axt.yaxis.set_minor_locator(AutoMinorLocator())
            axt.yaxis.set_major_locator(MultipleLocator(1.5))
            axt.set_ylabel("$[100\; {\\rm km/s}]$")

        axE.set_ylabel("$E(t) \;\;[M]$")
        for ax,d in zip(axJ,["x","y","z","J_k"]):
            ax.set_ylim(-0.1,0.5)
            ax.set_ylabel("$\mathbf{J}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")
        axP[0].text(0.05,0.7,'$q=0.5\,...\,1$\n$\chi_1=\chi_2=0$',transform=axP[0].transAxes,linespacing=1.4)

        #return [figP,figE,figJ]
        return figP

    @classmethod
    @plottingstuff
    def centerofmass(self):
        '''Fig. 2. Center of mass trajectories.
        Usage: surrkick.plots.centerofmass()'''

        allfig=[]

        # Left panel
        if True:

            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=0.5
            chi1=[0,0,0]
            chi2=[0,0,0]

            sk = surrkick(q=q , chi1=chi1,chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft[np.logical_and(sk.times>-4000,sk.times<25.5)])
            ax.plot(x-x0,y-y0,z-z0)
            ax.scatter(0,0,0,marker='.',s=60,alpha=0.5)
            x,y,z=np.transpose(sk.xoft)
            vx,vy,vz=np.transpose(sk.voft)

            for t in [-10,3,12,17,24]:
                i = np.abs(sk.times - t).argmin()
                v=np.linalg.norm([vx[i],vy[i],vz[i]])
                arrowsize=1e-4
                ax.quiver(x[i]-x0,y[i]-y0,z[i]-z0,vx[i]*arrowsize/v,vy[i]*arrowsize/v,vz[i]*arrowsize/v,length=0.0001,arrow_length_ratio=50000,alpha=0.5)

            ax.set_xlim(-0.004,0.0045)
            ax.set_ylim(-0.0025,0.006)
            ax.set_zlim(-0.006,0.0035)
            ax.set_xticklabels(ax.get_xticks(), fontsize=9)
            ax.set_yticklabels(ax.get_yticks(), fontsize=9)
            ax.set_zticklabels(ax.get_zticks(), fontsize=9)
            fig.text(0.38,0.45,'$q='+str(q)+'$\n$\\chi_1=0$\n${\\chi_2}=0$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
            ax.set_xlabel("$x\;\;[M]$",fontsize=13)
            ax.set_ylabel("$y\;\;[M]$",fontsize=13)
            ax.set_zlabel("$z\;\;[M]$",fontsize=13)
            ax.xaxis.labelpad=6
            ax.yaxis.labelpad=8
            ax.zaxis.labelpad=4
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.zaxis.set_minor_locator(AutoMinorLocator())

            allfig.append(fig)

        # Middle panel
        if True:

            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=0.5
            chi1=[0.8,0,0]
            chi2=[-0.8,0,0]

            sk = surrkick(q=q , chi1=chi1, chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft[np.logical_and(sk.times>-3500,sk.times<16)])
            ax.plot(x-x0,y-y0,z-z0)
            ax.scatter(0,0,0,marker='.',s=40,alpha=0.5)
            x,y,z=np.transpose(sk.xoft)
            vx,vy,vz=np.transpose(sk.voft)

            for t in [-20,-3,3,3,10,14]:
                i = np.abs(sk.times - t).argmin()
                v=np.linalg.norm([vx[i],vy[i],vz[i]])
                arrowsize=2e-3
                ax.quiver(x[i]-x0,y[i]-y0,z[i]-z0,vx[i]*arrowsize/v,vy[i]*arrowsize/v,vz[i]*arrowsize/v,length=0.00001,arrow_length_ratio=30000,alpha=0.5)

            ax.set_xlim(-0.005,0.005)
            ax.set_ylim(-0.005,0.005)
            ax.set_zlim(-0.005,0.005)
            ax.set_xticklabels(ax.get_xticks(), fontsize=9)
            ax.set_yticklabels(ax.get_yticks(), fontsize=9)
            ax.set_zticklabels(ax.get_zticks(), fontsize=9)
            fig.text(0.1,0.4,'$q='+str(q)+'$\n$\\boldsymbol{\\chi_1}=[0.8,0,0]$\n$\\boldsymbol{\\chi_2}=[-0.8,0,0]$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
            ax.set_xlabel("$x\;\;[M]$",fontsize=13)
            ax.set_ylabel("$y\;\;[M]$",fontsize=13)
            ax.set_zlabel("$z\;\;[M]$",fontsize=13)
            ax.xaxis.labelpad=6
            ax.yaxis.labelpad=8
            ax.zaxis.labelpad=4
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.zaxis.set_minor_locator(AutoMinorLocator())

            allfig.append(fig)

        # Right panel
        if True:

            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=1
            chi1=[0.81616392, 0.01773234, 0.57754829]
            chi1=np.array(chi1)*0.8/np.linalg.norm(chi1)
            chi2=[-0.87810809, 0.06485156, 0.47404689]
            chi2=np.array(chi2)*0.8/np.linalg.norm(chi2)

            sk = surrkick(q=q , chi1=chi1, chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft[np.logical_and(sk.times>-4500,sk.times<7)])
            ax.plot(x-x0,y-y0,z-z0)
            ax.scatter(0,0,0,marker='.',s=40,alpha=0.5)
            x,y,z=np.transpose(sk.xoft)
            vx,vy,vz=np.transpose(sk.voft)

            for t in [-9,3,4.5]:
                i = np.abs(sk.times - t).argmin()
                v=np.linalg.norm([vx[i],vy[i],vz[i]])
                arrowsize=2e-3
                ax.quiver(x[i]-x0,y[i]-y0,z[i]-z0,vx[i]*arrowsize/v,vy[i]*arrowsize/v,vz[i]*arrowsize/v,length=0.00001,arrow_length_ratio=30000,alpha=0.5)

            ax.set_xlim(-0.006,0.005)
            ax.set_ylim(-0.006,0.005)
            ax.set_zlim(0,0.011)
            ax.set_xticklabels(ax.get_xticks(), fontsize=9)
            ax.set_yticklabels(ax.get_yticks(), fontsize=9)
            ax.set_zticklabels(ax.get_zticks(), fontsize=9)
            fig.text(0.09,0.45,'$q='+str(q)+'$\n$\\boldsymbol{\\chi_1}=['+
                str(round(chi1[0],2))+','+str(round(chi1[1],2))+','+str(round(chi1[2],2))+
                ']$\n$\\boldsymbol{\\chi_2}=['+
                str(round(chi2[0],2))+','+str(round(chi2[1],2))+','+str(round(chi2[2],2))+
                ']$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
            ax.set_xlabel("$x\;\;[M]$",fontsize=13)
            ax.set_ylabel("$y\;\;[M]$",fontsize=13)
            ax.set_zlabel("$z\;\;[M]$",fontsize=13)
            ax.xaxis.labelpad=6
            ax.yaxis.labelpad=8
            ax.zaxis.labelpad=4

            allfig.append(fig)

        return allfig

    @classmethod
    @animate
    def recoil(self):
        '''Animated version of Fig. 2. Center of mass trajectories.
        Usage: surrkick.plots.recoil()'''

        leftpanel,middlepanel,rightpanel=True,True,True
        #leftpanel,middlepanel,rightpanel=True,False,False
        #leftpanel,middlepanel,rightpanel=False,True,False
        #leftpanel,middlepanel,rightpanel=False,False,True
        allfig=[]

        tnew=np.linspace(-4500,100,4601)
        tnew=np.append(tnew,np.ones(100)*tnew[-1])
        #tnew=tnew[::100]

        # Left panel
        if leftpanel:

            q=0.5
            chi1=[0,0,0]
            chi2=[0,0,0]
            sk = surrkick(q=q , chi1=chi1,chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft)
            xnew=spline(sk.times,x)(tnew)
            ynew=spline(sk.times,y)(tnew)
            znew=spline(sk.times,z)(tnew)
            tmax=100 # Fix annyoing bug in matplotlib.axes3d (comes from a 2d backend)
            temp=xnew[tnew<tmax]
            xnew=np.append(temp,temp[-1]*np.ones(len(xnew[tnew>=tmax])))
            temp=ynew[tnew<tmax]
            ynew=np.append(temp,temp[-1]*np.ones(len(ynew[tnew>=tmax])))
            temp=znew[tnew<tmax]
            znew=np.append(temp,temp[-1]*np.ones(len(znew[tnew>=tmax])))

            def _recoil(tilltime):
                fig = plt.figure(figsize=(6,6))
                ax=fig.add_axes([0,0,0.7,0.7], projection='3d')
                fig.text(0.25,0.67,'$t='+str(int(tilltime))+'M$',transform=fig.transFigure, horizontalalignment='left',verticalalignment='bottom')
                x=xnew[tnew<tilltime]
                y=ynew[tnew<tilltime]
                z=znew[tnew<tilltime]
                ax.plot(x-x0,y-y0,z-z0)
                if tilltime>0:
                    ax.scatter(0,0,0,marker='.',s=60,alpha=0.5)
                x,y,z=np.transpose(sk.xoft)
                vx,vy,vz=np.transpose(sk.voft)
                ax.set_xlim(-0.004,0.0045)
                ax.set_ylim(-0.0025,0.006)
                ax.set_zlim(-0.006,0.0035)
                ax.set_xticklabels(ax.get_xticks(), fontsize=9)
                ax.set_yticklabels(ax.get_yticks(), fontsize=9)
                ax.set_zticklabels(ax.get_zticks(), fontsize=9)
                fig.text(0.38,0.45,'$q='+str(q)+'$\n$\\chi_1=0$\n${\\chi_2}=0$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
                ax.set_xlabel("$x\;\;[M]$",fontsize=13)
                ax.set_ylabel("$y\;\;[M]$",fontsize=13)
                ax.set_zlabel("$z\;\;[M]$",fontsize=13)
                ax.xaxis.labelpad=6
                ax.yaxis.labelpad=8
                ax.zaxis.labelpad=4
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.zaxis.set_minor_locator(AutoMinorLocator())
                return fig

            figs=[_recoil(t) for t in tqdm(tnew)]
            allfig.append(figs)

        else:
            allfig.append([None])

        # Middle panel
        if middlepanel:

            q=0.5
            chi1=[0.8,0,0]
            chi2=[-0.8,0,0]
            sk = surrkick(q=q , chi1=chi1,chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft)
            xnew=spline(sk.times,x)(tnew)
            ynew=spline(sk.times,y)(tnew)
            znew=spline(sk.times,z)(tnew)
            tmax=50 # Fix annyoing bug in matplotlib.axes3d (comes from a 2d backend)
            temp=xnew[tnew<tmax]
            xnew=np.append(temp,temp[-1]*np.ones(len(xnew[tnew>=tmax])))
            temp=ynew[tnew<tmax]
            ynew=np.append(temp,temp[-1]*np.ones(len(ynew[tnew>=tmax])))
            temp=znew[tnew<tmax]
            znew=np.append(temp,temp[-1]*np.ones(len(znew[tnew>=tmax])))

            def _recoil(tilltime):
                fig = plt.figure(figsize=(6,6))
                ax=fig.add_axes([0,0,0.7,0.7], projection='3d')
                fig.text(0.25,0.67,'$t='+str(int(tilltime))+'M$',transform=fig.transFigure, horizontalalignment='left',verticalalignment='bottom')
                x=xnew[tnew<tilltime]
                y=ynew[tnew<tilltime]
                z=znew[tnew<tilltime]
                ax.plot(x-x0,y-y0,z-z0)
                if tilltime>0:
                    ax.scatter(0,0,0,marker='.',s=60,alpha=0.5)
                x,y,z=np.transpose(sk.xoft)
                vx,vy,vz=np.transpose(sk.voft)
                ax.set_xlim(-0.005,0.005)
                ax.set_ylim(-0.005,0.005)
                ax.set_zlim(-0.005,0.005)
                ax.set_xticklabels(ax.get_xticks(), fontsize=9)
                ax.set_yticklabels(ax.get_yticks(), fontsize=9)
                ax.set_zticklabels(ax.get_zticks(), fontsize=9)
                fig.text(0.1,0.4,'$q='+str(q)+'$\n$\\boldsymbol{\\chi_1}=[0.8,0,0]$\n$\\boldsymbol{\\chi_2}=[-0.8,0,0]$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
                ax.set_xlabel("$x\;\;[M]$",fontsize=13)
                ax.set_ylabel("$y\;\;[M]$",fontsize=13)
                ax.set_zlabel("$z\;\;[M]$",fontsize=13)
                ax.xaxis.labelpad=6
                ax.yaxis.labelpad=8
                ax.zaxis.labelpad=4
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.zaxis.set_minor_locator(AutoMinorLocator())
                return fig

            figs=[_recoil(t) for t in tqdm(tnew)]
            allfig.append(figs)

        else:
            allfig.append([None])

        # Right panel
        if rightpanel:

            q=1
            chi1=[0.81616392, 0.01773234, 0.57754829]
            chi1=np.array(chi1)*0.8/np.linalg.norm(chi1)
            chi2=[-0.87810809, 0.06485156, 0.47404689]
            chi2=np.array(chi2)*0.8/np.linalg.norm(chi2)
            sk = surrkick(q=q , chi1=chi1,chi2=chi2)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]
            x,y,z=np.transpose(sk.xoft)
            xnew=spline(sk.times,x)(tnew)
            ynew=spline(sk.times,y)(tnew)
            znew=spline(sk.times,z)(tnew)
            tmax=30 # Fix annyoing bug in matplotlib.axes3d (comes from a 2d backend)
            temp=xnew[tnew<tmax]
            xnew=np.append(temp,temp[-1]*np.ones(len(xnew[tnew>=tmax])))
            temp=ynew[tnew<tmax]
            ynew=np.append(temp,temp[-1]*np.ones(len(ynew[tnew>=tmax])))
            temp=znew[tnew<tmax]
            znew=np.append(temp,temp[-1]*np.ones(len(znew[tnew>=tmax])))

            def _recoil(tilltime):
                fig = plt.figure(figsize=(6,6))
                ax=fig.add_axes([0,0,0.7,0.7], projection='3d')
                fig.text(0.25,0.67,'$t='+str(int(tilltime))+'M$',transform=fig.transFigure, horizontalalignment='left',verticalalignment='bottom')
                x=xnew[tnew<tilltime]
                y=ynew[tnew<tilltime]
                z=znew[tnew<tilltime]
                ax.plot(x-x0,y-y0,z-z0)
                if tilltime>0:
                    ax.scatter(0,0,0,marker='.',s=60,alpha=0.5)
                x,y,z=np.transpose(sk.xoft)
                vx,vy,vz=np.transpose(sk.voft)
                ax.set_xlim(-0.006,0.005)
                ax.set_ylim(-0.006,0.005)
                ax.set_zlim(0,0.011)
                ax.set_xticklabels(ax.get_xticks(), fontsize=9)
                ax.set_yticklabels(ax.get_yticks(), fontsize=9)
                ax.set_zticklabels(ax.get_zticks(), fontsize=9)
                fig.text(0.09,0.45,'$q='+str(q)+'$\n$\\boldsymbol{\\chi_1}=['+
                    str(round(chi1[0],2))+','+str(round(chi1[1],2))+','+str(round(chi1[2],2))+
                    ']$\n$\\boldsymbol{\\chi_2}=['+
                    str(round(chi2[0],2))+','+str(round(chi2[1],2))+','+str(round(chi2[2],2))+
                    ']$\n$v_k='+str(int(convert.kms(sk.kick)))+'{\\rm \;km/s}$',transform=fig.transFigure)
                ax.set_xlabel("$x\;\;[M]$",fontsize=13)
                ax.set_ylabel("$y\;\;[M]$",fontsize=13)
                ax.set_zlabel("$z\;\;[M]$",fontsize=13)
                ax.xaxis.labelpad=6
                ax.yaxis.labelpad=8
                ax.zaxis.labelpad=4
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.zaxis.set_minor_locator(AutoMinorLocator())
                return fig

            figs=[_recoil(t) for t in tqdm(tnew)]
            allfig.append(figs)

        else:
            allfig.append([None])

        return allfig

    @classmethod
    @plottingstuff
    def hangupErad(self):
        '''Fig. 3. Energy radiated for binaries with aligned spins.
        Usage: surrkick.plots.hangupErad()'''

        figs=[]
        L=0.7
        H=0.5
        S=0.05
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,L,H])

        q=0.5
        chimag=0.8
        sks=[ surrkick(q=q,chi1=[0,0,chimag],chi2=[0,0,chimag],t_ref=-4500),
            surrkick(q=q,chi1=[0,0,-chimag],chi2=[0,0,-chimag],t_ref=-4500),
            surrkick(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag],t_ref=-4500),
            surrkick(q=q, chi1=[0,0,-chimag],chi2=[0,0,chimag],t_ref=-4500),
            surrkick(q=q, chi1=[0,0,0],chi2=[0,0,0],t_ref=-4500)]

        fks=[precession.finalmass(0,0,0,q,chimag/(1+q)**2,chimag*(q/(1+q))**2),
            precession.finalmass(np.pi,np.pi,0,q,chimag/(1+q)**2,chimag*(q/(1+q))**2),
            precession.finalmass(0,np.pi,0,q,chimag/(1+q)**2,chimag*(q/(1+q))**2),
            precession.finalmass(np.pi,0,0,q,chimag/(1+q)**2,chimag*(q/(1+q))**2),
            precession.finalmass(0,0,0,q,0,0)]

        labels=["$\chi_i\!=\!0.8$, up-up","$\chi_i\!=\!0.8$, down-down","$\chi_i\!=\!0.8$, up-down","$\chi_i\!=\!0.8$, down-up","$\chi_i=0$"]
        dashes=['',[15,5],[8,5],[2,2],[0.5,1]]
        cols=['C0','C1','C2','C3','black']
        for sk,fk,l,d,c in tqdm(zip(sks,fks,labels,dashes,cols)):
            ax.plot(sk.times,sk.Eoft,alpha=0.4,lw=1,c=c)
            ax.plot(sk.times,sk.Eoft,alpha=1,lw=2,c=c,dashes=d,label=l)
                #ax.axhline((1-fk)*(1+sk.Eoft[0]),c=c,dashes=d)

        ax.legend(loc="upper left",fontsize=11,handlelength=5.5)
        ax.text(0.8,0.1,'$q='+str(q)+'$',transform=ax.transAxes,linespacing=1.4)
        ax.set_xlim(-50,50)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("$t\;\;[M]$")
        ax.set_ylim(-0.0,0.08)
        ax.set_ylabel("${E(t)} \;\;[M]$")

        return fig

    @classmethod
    @plottingstuff
    def spinaligned(self):
        '''Fig. 4. Kicks for binaries with aligned spins.
        Usage: surrkick.plots.spinaligned()'''

        figs=[]
        L=0.7
        H=0.3
        S=0.05

        for q in tqdm([1,0.5]):

            fig = plt.figure(figsize=(6,6))
            axs = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
            axst = [ax.twinx() for ax in axs]

            chimag=0.8
            sks=[ surrkick(q=q,chi1=[0,0,chimag],chi2=[0,0,chimag],t_ref=-100), surrkick(q=q,chi1=[0,0,-chimag],chi2=[0,0,-chimag],t_ref=-100),
            surrkick(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag],t_ref=-100),
            surrkick(q=q, chi1=[0,0,-chimag],chi2=[0,0,chimag],t_ref=-100)]
            labels=["up-up","down-down","up-down","down-up"]
            dashes=['',[15,5],[8,5],[2,2]]
            cols=['C0','C1','C2','C3']
            for sk,l,d,c in tqdm(zip(sks,labels,dashes,cols)):
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=0.4,lw=1,c=c)
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=0.4,lw=1,c=c)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=0.4,lw=1,c=c)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=1,lw=2,c=c,dashes=d,label=l)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=0.4,lw=1,c=c)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=1,lw=2,c=c,dashes=d)

            axs[2].legend(loc="lower left",fontsize=14,ncol=2,handlelength=3.86)
            axs[0].text(0.05,0.7,'$q='+str(q)+'$\n$\chi_1=\chi_2=0.8$',transform=axs[    0].transAxes,linespacing=1.4)
            for ax in axs:
                ax.set_xlim(-50,50)
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            for ax in axs[:-1]:
                ax.set_xticklabels([])
            for ax in [axs[-1]]:
                ax.set_xlabel("$t\;\;[M]$")
            for ax,d in zip(axs,["x","y","z","v_k"]):
                ax.set_ylim(-1.5,1.5)
                ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")
            for axt in axst:
                axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/100)
                axt.xaxis.set_minor_locator(AutoMinorLocator())
                axt.yaxis.set_minor_locator(AutoMinorLocator())
                axt.yaxis.set_major_locator(MultipleLocator(2))
                axt.set_ylabel("$[100\;{\\rm km/s}]$")

            figs.append(fig)

        return figs

    @classmethod
    @plottingstuff
    def leftright(self):
        '''Fig. 5. Kicks for binaries with spins in the orbital plane.
        Usage: surrkick.plots.leftright()'''

        figs=[]
        L=0.7
        H=0.3
        S=0.05
        Z=0.35

        for q in tqdm([1,0.5]):

            fig = plt.figure(figsize=(6,6))
            axs = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
            axi = fig.add_axes([0,-1*(S+H)-S-Z*H,Z*L*1.2,Z*H])
            axst = [ax.twinx() for ax in axs]

            chimag1=0.8
            chimag2=0.8
            sks=[ surrkick(q=q,chi1=[chimag1,0,0],chi2=[chimag2,0,0],t_ref=-125),
            surrkick(q=q,chi1=[-chimag1,0,0],chi2=[-chimag2,0,0],t_ref=-125),
            surrkick(q=q,chi1=[chimag1,0,0],chi2=[-chimag2,0,0],t_ref=-125),
            surrkick(q=q, chi1=[-chimag1,0,0],chi2=[chimag2,0,0],t_ref=-125)]
            labels=["right-right","left-left","right-left","left-right"]
            dashes=['',[15,5],[8,5],[2,2]]
            cols=['C0','C1','C2','C3']
            for sk,l,d,c in tqdm(zip(sks,labels,dashes,cols)):
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=0.4,lw=1,c=c)
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=0.4,lw=1,c=c)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=1,lw=2,c=c,dashes=d,label=l)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=0.4,lw=1,c=c)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=1,lw=2,c=c,dashes=d)
                axi.plot(sk.times,convert.kms(project(sk.voft,[0,0,1])),alpha=0.4,lw=1,c=c)
                axi.plot(sk.times,convert.kms(project(sk.voft,[0,0,1])),alpha=1,lw=2,c=c,dashes=d)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=0.4,lw=1,c=c)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=1,lw=2,c=c,dashes=d)

            axs[1].legend(loc="lower left",fontsize=14,ncol=2,handlelength=3.86)
            axs[0].text(0.05,0.7,'$q='+str(q)+'$\n$\chi_1=\chi_2=0.8$',transform=axs[    0].transAxes,linespacing=1.4)
            for ax in axs:
                ax.set_xlim(-50,50)
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            for ax in axs[:-1]:
                ax.set_xticklabels([])
            for ax in [axs[-1]]:
                ax.set_xlabel("$t\;\;[M]$")
            for ax,d in zip(axs,["x","y","z","v_k"]):
                ax.set_ylim(-10,10)
                ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")
            for axt in axst:
                axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/1000)
                axt.xaxis.set_minor_locator(AutoMinorLocator())
                axt.yaxis.set_minor_locator(AutoMinorLocator())
                axt.yaxis.set_major_locator(MultipleLocator(1))
                axt.set_ylabel("$[1000\;{\\rm km/s}]$")
            axi.set_xlim(-450,-150)
            axi.set_ylim(-7,7)
            axi.set_ylabel("$[{\\rm km/s}]$",fontsize=8,labelpad=1)
            axi.yaxis.tick_right()
            axi.yaxis.set_label_position("right")

            axi.set_xticks([-400,-300,-200])
            axi.set_yticks([-5,0,5])
            axi.set_yticklabels(axi.get_yticks(),fontsize=8)
            axi.xaxis.set_tick_params(pad=1)
            axi.set_xticklabels(axi.get_xticks(),fontsize=8)
            axi.yaxis.set_tick_params(pad=1)
            axi.yaxis.set_ticks_position('right')
            axi.xaxis.set_ticks_position('bottom')

            figs.append(fig)

        return figs

    @classmethod
    @plottingstuff
    def alphaseries(self):
        '''Fig. 6. Degeneracy between the reference time and an overall spin rotation.
        Usage: surrkick.plots.alphaseries()'''

        fig = plt.figure(figsize=(6,6))
        L=0.7
        H=0.3
        S=0.05
        axs = [fig.add_axes([i*(S+L),0,L,H]) for i in [0,1]]
        axt = axs[1].twinx()

        dim=200
        chimag=0.8
        tref_vals=np.linspace(-250,-100,dim)
        kick_vals=[]
        for t_ref in tqdm(tref_vals):
            sk = surrkick(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=t_ref)
            kick_vals.append(sk.kick)

        axs[0].plot(tref_vals,1/0.001*np.array(kick_vals))
        axs[0].scatter(-125,1/0.001*surrkick(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=-125).kick,marker='o',edgecolor='C1',facecolor='none',s=100,linewidth='2')
        axs[0].scatter(-100,1/0.001*surrkick(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=-100).kick,marker='x',edgecolor='gray',facecolor='gray',s=100,linewidth='2')
        alpha_vals=np.linspace(-np.pi,np.pi,dim)
        kick_vals=[]
        for alpha in tqdm(alpha_vals):
            sk = surrkick(q=1 , chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])
            kick_vals.append(sk.kick)
        axs[1].plot(alpha_vals,1/0.001*np.array(kick_vals),c='C3')
        axs[1].scatter(0,1/0.001*surrkick(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=-100).kick,marker='x',edgecolor='gray',facecolor='gray',s=100,linewidth='2')

        axs[0].text(0.05,0.5,'$q=1$\n$\chi_1=\chi_2=0.8$\n$\\alpha=0$',transform=axs[0].transAxes,linespacing=1.4)
        axs[1].text(0.05,0.5,'$q=1$\n$\chi_1=\chi_2=0.8$\n$t_{\\rm ref}=-100M$',transform=axs[1].transAxes,linespacing=1.4)
        axs[1].set_yticklabels([])
        axs[0].set_xlabel("$t_{\\rm ref}\;\;[M]$")
        axs[1].set_xlim(-1.1*np.pi,1.1*np.pi)
        axs[1].set_xlabel("$\\alpha$")
        axs[1].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        axs[1].set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
        axs[0].set_ylabel("${v_k} \;\;[0.001 c]$")
        for ax in axs:
            ax.set_ylim(0,10)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/1000)
        axt.xaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_major_locator(MultipleLocator(1))
        axt.set_ylabel("$[1000\;{\\rm km/s}]$")

        return fig

    @classmethod
    @plottingstuff
    def alphaprof(self):
        '''Fig. 7. Role of the orbital phase at merger.
        Usage: surrkick.plots.alphaprof()'''

        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.3])
        axt = ax.twinx()
        chimag=0.5

        ax.axvline(0,c='black',alpha=0.3,ls='dotted')
        ax.axhline(0,c='black',alpha=0.3,ls='dotted')
        alpha_vals=np.linspace(-np.pi,np.pi,30)
        kick_vals=[]
        for i, alpha in enumerate(tqdm(alpha_vals)):
            sk = surrkick(q=1, chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])
            color=plt.cm.copper(i/len(alpha_vals))
            ax.plot(sk.times,1/0.001*project(sk.voft,sk.kickdir),color=color,alpha=0.8)
        ax.set_xlim(-100,50)
        ax.set_ylim(-1.2,6.7)

        ax.set_xlabel("$t\;\;[M]$")
        ax.set_ylabel("$\mathbf{v}(t)\cdot \mathbf{\hat v_k} \;\;[0.001 c]$")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.text(0.05,0.55,'$q=1$\n$\chi_1=\chi_2=0.8$\n$\\alpha=-\pi ... \pi$',transform=ax.transAxes,linespacing=1.4)
        axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/1000)
        axt.xaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_major_locator(MultipleLocator(0.5))
        axt.set_ylabel("$[1000\;{\\rm km/s}]$")

        return fig

    @classmethod
    def findlarge(self):
        '''Generate large sample of binaries to find hang-up kicks (no plot).
        Usage: surrkick.plots.findlarge()'''

        dim=int(1e5)
        filename='findlarge.pkl'

        q=1
        chi1=0.8
        chi2=0.8

        if not os.path.isfile(filename):

            def _kickdistr(i):
                np.random.seed()
                phi1 = np.random.uniform(0,2*np.pi)
                theta1 = np.arccos(np.random.uniform(-1,1))
                chi1v= [ chi1*np.sin(theta1)*np.cos(phi1), chi1*np.sin(theta1)*np.sin(phi1), chi1*np.cos(theta1) ]
                phi2 = np.random.uniform(0,2*np.pi)
                theta2 = np.arccos(np.random.uniform(-1,1))
                chi2v= [ chi2*np.sin(theta2)*np.cos(phi2), chi2*np.sin(theta2)*np.sin(phi2), chi2*np.cos(theta2) ]
                sk= surrkick(q=q,chi1=chi1v,chi2=chi2v)

                dummy,dummy,dummy,S1,S2=precession.get_fixed(q,chi1,chi2)
                fk=precession.finalkick(theta1,theta2,phi2-phi1,1,S1,S2,maxkick=False,kms=False,more=False)

                return [sk.kick,fk,theta1,theta2]

            print("Running in parallel on", multiprocessing.cpu_count(),"cores. Storing data:", filename)
            parmap = pathos.multiprocessing.ProcessingPool(multiprocessing.cpu_count()).imap
            data= list(tqdm(parmap(_kickdistr, range(dim)),total=dim))

            with open(filename, 'wb') as f: pickle.dump(zip(*data), f)
        with open(filename, 'rb') as f: skicks,fkicks,theta1,theta2 = pickle.load(f)


        print("Largest kick (surrogate):", convert.kms(max(skicks)))
        maxsk= skicks==max(skicks)
        print('theta1=',np.degrees(np.array(theta1)[maxsk][0]))
        print('theta2=',np.degrees(np.array(theta2)[maxsk][0]))
        print("Largest kick (fitting formula):", convert.kms(max(fkicks)))

        return []

    @classmethod
    @plottingstuff
    def lineofsight(self):
        '''Fig. 8. Projections of the kick profile.
        Usage: surrkick.plots.lineofsight()'''

        L=0.7
        H=0.6
        S=0.05
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,L,H])
        axt = ax.twinx()
        ax.axhline(0,c='black',alpha=0.3,ls='dotted')

        q=0.5
        chimag=0.8
        sk= surrkick(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag],t_ref=-100)
        dim=15

        filename='lineofsight.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)

            store=[]
            for i in tqdm(np.linspace(0,1,dim)):
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                randomvec= [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
                store.append([randomvec,project(sk.voft,randomvec)[-1]])

            store=sorted(store, key=lambda x:x[1])
            with open(filename, 'wb') as f: pickle.dump(store, f)
        with open(filename, 'rb') as f: store = pickle.load(f)

        for i,rv in tqdm(zip(np.linspace(0,1,dim),[x[0] for x in store])):
            color=plt.cm.copper(i)
            ax.plot(sk.times,1./0.001*project(sk.voft,rv),c=color,alpha=0.8)

        ax.text(0.05,0.75,'$q='+str(q)+'$\n$\chi_1=\chi_2=0.8$\n right-left',transform=ax.transAxes,linespacing=1.4)
        ax.set_xlim(-50,50)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("$t\;\;[M]$")
        ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat n} \;\;[0.001c]$")
        axt.set_ylim(convert.kms(1e-3*np.array(ax.get_ylim()))/100)
        axt.xaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_major_locator(MultipleLocator(1))
        axt.set_ylabel("$[100\;{\\rm km/s}]$")

        return fig

    @classmethod
    @plottingstuff
    def explore(self):
        '''Fig. 9. Radiated quantities on statistical sample of binaries, compare with fitting formula.
        Usage: surrkick.plots.explore()'''

        fig = plt.figure(figsize=(6,6))
        L=0.6
        H=0.7
        S=0.12
        s=0.04
        Li=0.35
        Hi=0.25
        axv= fig.add_axes([0,0,H,H])
        axE= fig.add_axes([H+S-0.02,(H-S)/2+S,L,(H-S)/2])
        axJ= fig.add_axes([H+S-0.02,0,L,(H-S)/2])
        axi = fig.add_axes([H-Li-s,H-Hi-0.15,Li,Hi])
        axt = axv.twiny()

        dim=int(1e6)

        filename='explore.pkl'
        if not os.path.isfile(filename):

            def _explore(i):
                #print(' ',i)
                np.random.seed()
                q=np.random.uniform(0.5,1)
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                sk= surrkick(q=q,chi1=chi1,chi2=chi2)
                q=np.random.uniform(0.5,1)
                chi1m = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi2m = 0.8*(np.random.uniform(0,1))**(1./3.)
                theta1=np.arccos(np.random.uniform(-1,1))
                theta2=np.arccos(np.random.uniform(-1,1))
                deltaphi = np.random.uniform(0,2*np.pi)
                dummy,dummy,dummy,S1,S2=precession.get_fixed(q,chi1m,chi2m)
                fk=precession.finalkick(theta1,theta2,deltaphi,q,S1,S2,maxkick=False,kms=False,more=False)
                fe=(1-precession.finalmass(theta1,theta2,deltaphi,q,S1,S2))*(1+sk.Eoft[0])
                return [sk.Erad,sk.kick,sk.Jrad,fk,fe]

            print("Running in parallel on", multiprocessing.cpu_count(),"cores. Storing data:", filename)
            parmap = pathos.multiprocessing.ProcessingPool(multiprocessing.cpu_count()).imap
            data= list(tqdm(parmap(_explore, range(dim)),total=dim))
            #data= map(_explore, range(dim))

            with open(filename, 'wb') as f: pickle.dump(zip(*data), f)
        with open(filename, 'rb') as f: Erad,kicks,Jrad,fk,fe = pickle.load(f)
        kicks=np.array(kicks)
        fk=np.array(fk)
        fe=np.array(fe)

        nbins=100
        axE.hist(fe,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='step',lw=2,alpha=0.8,color='C3',normed=True)
        axE.hist(fe,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='stepfilled',alpha=0.2,color='C3',normed=True)
        axE.hist(Erad,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='step',lw=2,alpha=0.8,color='C0',normed=True)
        axE.hist(Erad,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='stepfilled',alpha=0.2,color='C0',normed=True)
        #print("surr", np.median(Erad), np.std(Erad))
        #print("fit", np.median(fe), np.std(fe))

        for ax in [axv,axi]:
            ax.hist(1/0.001*fk,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='step',lw=2,alpha=0.8,color='C3',label='Fitting formula',normed=True)
            ax.hist(1/0.001*fk,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='stepfilled',alpha=0.2,color='C3',normed=True)
            ax.hist(1/0.001*kicks,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='step',lw=2,alpha=0.8,color='C0',label='Surrogate',normed=True)
            ax.hist(1/0.001*kicks,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='stepfilled',alpha=0.2,color='C0',normed=True)
        axJ.hist(Jrad,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='step',lw=2,alpha=0.8,color='C0',normed=True)
        axJ.hist(Jrad,bins=nbins,weights=np.ones_like(Erad)/dim,histtype='stepfilled',alpha=0.2,color='C0',normed=True)

        axE.set_xlabel("$E \;\;[M]$")
        axv.set_xlabel("$v_k\;\;[0.001c]$")
        axJ.set_xlabel("$J\;\;[M^2]$")
        for ax in [axE,axv,axJ]:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        axv.xaxis.set_minor_locator(MultipleLocator(0.5))
        axv.xaxis.set_major_locator(MultipleLocator(2))
        axv.legend(loc='upper right',fontsize=15)
        axE.set_xlim(0.015,0.085)
        axE.set_ylim(0,60)
        axv.set_xlim(-0.5,12.5)
        axv.set_ylim(0,0.35)
        axJ.set_ylim(0,8)
        axi.set_xlim(8,11)
        axi.set_ylim(0,0.01)
        axJ.yaxis.set_major_locator(MultipleLocator(2))
        axt.set_xlim(convert.kms(1e-3*np.array(axv.get_xlim())))
        axt.xaxis.set_minor_locator(AutoMinorLocator())
        axt.yaxis.set_minor_locator(AutoMinorLocator())
        axt.xaxis.set_major_locator(MultipleLocator(1000))
        axt.set_xlabel("$[{\\rm km/s}]$",labelpad=8)
        axi.xaxis.set_major_locator(MultipleLocator(1))
        axi.yaxis.set_major_locator(MultipleLocator(0.002))
        axi.set_yticklabels(axi.get_yticks(),fontsize=12)
        axi.set_xticklabels(axi.get_xticks(),fontsize=12)
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(axv, axi, loc1=4, loc2=3, fc="none", ec="0.5",alpha=0.8)

        print("Largest kicks.\n\tSurrogate:", convert.kms(max(kicks)), "\n\tFitting formula", convert.kms(max(fk)))
        print("P(vk>2000 km/s).\n\tSurrogate:", len(kicks[kicks>convert.cisone(2000)])/dim, "\n\tFitting formula", len(fk[fk>convert.cisone(2000)])/dim)

        return fig

    @classmethod
    @plottingstuff
    def normprofiles(self):
        '''Fig. 10. Shape of the kicks.
        Usage: surrkick.plots.normprofiles()'''

        levels = np.linspace(0,1.6,100)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=plt.cm.copper,extend='max')
        plt.clf()

        fig = plt.figure(figsize=(4,4))
        ax=fig.add_axes([0,0,0.7,0.7])
        ax.axhline(0,c='black',alpha=0.3,ls='dotted')
        ax.axhline(1,c='black',alpha=0.3,ls='dotted')

        filename='normprofiles.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)

            data=[]
            for i in tqdm(range(200)):
                q=np.random.uniform(0.5,1)
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                sk= surrkick(q=q,chi1=chi1,chi2=chi2)
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                randomdir= [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ]
                data.append([sk.kick, project(sk.voft/sk.kick,sk.kickdir)])

            with open(filename, 'wb') as f: pickle.dump(data, f)
        with open(filename, 'rb') as f: data = pickle.load(f)

        times=surrkick().times
        #data.sort(key=lambda x: x[0])
        for d in tqdm(data):
            ax.plot(times,d[1],alpha=0.7, c= plt.cm.copper(d[0]/0.0016),lw=1)

        axcb = fig.add_axes([0.72,0,0.05,0.7])
        cb = fig.colorbar(CS3,cax=axcb,boundaries=np.linspace(0,1.6,100),ticks=np.linspace(0,1.6,9))
        ax.plot(times,scipy.stats.norm.cdf(times, loc=10, scale=8),dashes=[10,4],c='C0',lw=2)
        ax.plot(times,scipy.stats.norm.cdf(times, loc=10, scale=8),c='C0',alpha=0.5,lw=1)
        ax.set_xlim(-50,50)
        ax.set_ylim(-2,3)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        cb.set_label('$v_k\;\;[0.001c]$')
        ax.set_xlabel("$t\;\;[M]$")
        ax.set_ylabel("$\mathbf{v}(t)\cdot\mathbf{\hat n}\, /\, \mathbf{v_k}\cdot\mathbf{\hat n}$")

        return fig

    @classmethod
    @plottingstuff
    def symmetry(self):
        '''Fig. 11. Exploit symmetries to test accuracy.
        Usage: surrkick.plots.symmetry()'''

        L=0.7
        H=0.33
        S=0.23
        fig = plt.figure(figsize=(6,6))
        ax = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2]]
        axt = [axx.twiny() for axx in ax]

        dim=int(1e4)
        filename='symmetry.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)

            kicks=[[],[],[]]

            for i in tqdm(range(dim)):
                q=1
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1 = [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                chi2 = chi1
                kicks[0].append(surrkick(q=q,chi1=chi1,chi2=chi2).kick)

            for i in tqdm(range(dim)):
                q=np.random.uniform(0.5,1)
                chi1 = [ 0, 0, np.random.uniform(-0.8,0.8) ]
                chi2 = [ 0, 0, np.random.uniform(-0.8,0.8) ]
                kicks[1].append(surrkick(q=q,chi1=chi1,chi2=chi2).kickcomp[2])

            for i in tqdm(range(dim)):
                q=1
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1 = [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                chi2 = [-chi1[0],-chi1[1],chi1[2]]
                kicks[2].append(np.linalg.norm(np.cross(surrkick(q=q,chi1=chi1,chi2=chi2).kickcomp,[0,0,1])))

            with open(filename, 'wb') as f: pickle.dump(kicks, f)
        with open(filename, 'rb') as f: kicks = pickle.load(f)

        nbins=100
        for axx,kick in zip(ax,kicks):
            axx.hist(1/0.001*np.abs(kick),bins=nbins,histtype='step',lw=2,alpha=1,color='C4',normed=True)
            axx.hist(1/0.001*np.abs(kick),bins=nbins,histtype='stepfilled',alpha=0.3,color='C4',normed=True)
            #print(np.percentile(1/0.001*np.abs(kick), 50),np.percentile(1/0.001*np.abs(kick), 90))
            axx.axvline(np.percentile(1/0.001*np.abs(kick), 50),c='gray',ls='dashed')
            axx.axvline(np.percentile(1/0.001*np.abs(kick), 90),c='gray',ls='dotted')

        ax[0].set_xlabel("$v_k \;\;[0.001c]$")
        ax[1].set_xlabel("$|\mathbf{v_k}\cdot \mathbf{\hat z}| \;\;[0.001c]$")
        ax[2].set_xlabel("$|\mathbf{v_k}\\times \mathbf{\hat z}| \;\;[0.001c]$")
        ax[0].text(0.5,0.9,'$q=1$\n$\\boldsymbol{\\chi_1}=\\boldsymbol{\\chi_2}$\n$\longrightarrow v_k=0$',verticalalignment='top', transform=ax[0].transAxes,linespacing=1.6)
        ax[1].text(0.5,0.9,'${\\rm Generic}\; q$\n$\\boldsymbol{\\chi_1}\\times \mathbf{\hat z} = \\boldsymbol{\\chi_2}\\times \mathbf{\hat z}=0$\n$\longrightarrow \mathbf{v_k}\cdot\mathbf{\hat z}=0$',verticalalignment='top',transform=ax[1].transAxes,linespacing=1.6)
        ax[2].text(0.5,0.9,'$q=1$\n$\\boldsymbol{\\chi_1}\\cdot \mathbf{\hat z}=\\boldsymbol{\\chi_2}\\cdot \mathbf{\hat z}$\n$\\boldsymbol{\\chi_1}\\times \mathbf{\hat z} = -\\boldsymbol{\\chi_2}\\times \mathbf{\hat z}$\n$\longrightarrow \mathbf{v_k}\\times\mathbf{\hat z}=0$',verticalalignment='top',transform=ax[2].transAxes,linespacing=1.6)
        for axx,axxt in zip(ax,axt):
            axx.xaxis.set_minor_locator(AutoMinorLocator())
            axx.yaxis.set_minor_locator(AutoMinorLocator())
            axxt.set_xlim(convert.kms(1e-3*np.array(axx.get_xlim())))
            axxt.xaxis.set_minor_locator(AutoMinorLocator())
            axxt.yaxis.set_minor_locator(AutoMinorLocator())
            axxt.set_xlabel("$[{\\rm km/s}]$",labelpad=8)
        axt[0].xaxis.set_major_locator(MultipleLocator(50))
        axt[1].xaxis.set_major_locator(MultipleLocator(5))
        axt[2].xaxis.set_major_locator(MultipleLocator(5))

        return fig

    @classmethod
    @plottingstuff
    def nr_comparison_histograms(self):
        '''Fig. 12. Comparison with SpEC simulations, sources of error.
        Usage: surrkick.plots.nr_comparison_histograms()'''

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,1.25,0.6])
        axt = ax.twiny()

        nr100 = np.loadtxt(os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/nr_kicks_t100.dat")
        nr4500 = np.loadtxt(os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/nr_kicks_t4500.dat")

        def _nr_surr_comparison_data_helper(nr_data, t):
            kicks = []
            for d in nr_data:
                q = d[2]
                chi1 = [d[3], d[4], d[5]]
                chi2 = [d[6], d[7], d[8]]
                kicks.append(surrkick(q=q, chi1=chi1, chi2=chi2, t_ref=t).kick)
            return np.array(kicks)

        filename='nr_comparison_kicks_t100.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)
            surr_kicks = _nr_surr_comparison_data_helper(nr100, -100)
            with open(filename, 'wb') as f: pickle.dump(surr_kicks, f)
        with open(filename, 'rb') as f: surr100 = pickle.load(f)

        filename='nr_comparison_kicks_t4500.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)
            surr_kicks = _nr_surr_comparison_data_helper(nr4500, -4500)
            with open(filename, 'wb') as f: pickle.dump(surr_kicks, f)
        with open(filename, 'rb') as f: surr4500 = pickle.load(f)

        mag_nr = nr4500[:,12] / 0.001
        mag_nr_lev2 = nr4500[:,16] / 0.001
        mag_nr_lmax4 = nr4500[:,20] / 0.001
        mag_surr = surr4500[:] / 0.001
        mag_surr_t100 = surr100[:] / 0.001
        delta_nr_surr = np.fabs(mag_nr - mag_surr)
        delta_nr_levs = np.fabs(mag_nr - mag_nr_lev2)
        delta_nr_lmax = np.fabs(mag_nr - mag_nr_lmax4)
        delta_surr_times = np.fabs(mag_surr - mag_surr_t100)

        logbins = np.logspace(-6, 1.3, 50)
        ax.hist(mag_nr, bins=logbins, histtype='stepfilled', alpha=0.6, label="$v_k$ NR", color='C3')
        ax.hist(mag_surr, bins=logbins, histtype='stepfilled', alpha=0.6, label="$v_k$ Surrogate", color='C0')
        ax.hist(mag_nr, bins=logbins, histtype='step', alpha=0.8,color='C3',lw=1.8)
        ax.hist(mag_surr, bins=logbins, histtype='step',alpha=0.8, color='C0',lw=1.8)
        ax.hist(delta_nr_surr, bins=logbins, histtype='step', label="$\Delta v_k$ NR vs. Surrogate",color='black',ls='dashed',lw=2,zorder=20)
        #print(np.percentile(delta_nr_surr, 90))
        ax.hist(delta_nr_levs, bins=logbins, histtype='step', label="$\Delta v_k$ NR resolution", color='C1',lw=1.8)
        ax.hist(delta_surr_times, bins=logbins, histtype='step', label="$\Delta v_k$ Surr $t_{\\rm ref}/M\!=\!-100$ vs.  $\!-4500$",color='C2',lw=1.8)
        ax.hist(delta_nr_lmax, bins=logbins, histtype='step', label="$\Delta v_k$ NR $l_{\\rm max}\!=\!8$ vs. $4$",color='C4',lw=1.8)

        ax.legend(loc=2, ncol=1, fontsize=14).set_zorder(100)
        ax.set_xscale("log")
        ax.set_xlabel("$v_k\;\;[0.001c]$")
        ax.set_ylim(0,120)
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.xaxis.set_major_locator(LogLocator(numticks=8))
        ax.xaxis.set_minor_locator(LogLocator(numticks=8,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        axt.semilogx()
        axt.set_xlim(convert.kms(1e-3*np.array(ax.get_xlim())))
        axt.yaxis.set_minor_locator(AutoMinorLocator())
        axt.set_xlabel("$[{\\rm km/s}]$",labelpad=8)


        return fig

    @classmethod
    @plottingstuff
    def nr_comparison_scatter(self):
        '''Fig. 13. Comparison with SpEC simulations, percentiles.
        Usage: surrkick.plots.nr_comparison_scatter()'''

        main_w = 0.6
        main_h = 0.6
        hist_h = 0.15
        gap = 0.04

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, main_w, main_h])
        axt = fig.add_axes([0, main_h + gap, main_w, hist_h])
        axr = fig.add_axes([main_w + gap, 0, hist_h, main_h])

        nr4500 = np.loadtxt(os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/nr_kicks_t4500.dat")

        # duplicated from histogram plot
        def _nr_surr_comparison_data_helper(nr_data, t):
            kicks = []
            for d in nr_data:
                q = d[2]
                chi1 = [d[3], d[4], d[5]]
                chi2 = [d[6], d[7], d[8]]
                kicks.append(surrkick(q=q, chi1=chi1, chi2=chi2, t_ref=t).kick)
            return np.array(kicks)

        # duplicated from histogram plot
        filename='nr_comparison_kicks_t4500.pkl'
        if not os.path.isfile(filename):
            surr_kicks = _nr_surr_comparison_data_helper(nr4500, -4500)
            print("Storing data:", filename)
            with open(filename, 'wb') as f: pickle.dump(surr_kicks, f)
        with open(filename, 'rb') as f: surr4500 = pickle.load(f)

        mag_nr = nr4500[:,12] / 0.001
        mag_surr = surr4500[:] / 0.001
        diff = np.fabs(mag_nr - mag_surr)
        perc50 = np.percentile(diff, 50)
        perc90 = np.percentile(diff, 90)
        print("  50th percentile of kick magnitude diffs [0.001c]:", perc50)
        print("  90th percentile of kick magnitude diffs [0.001c]:", perc90)
        x = np.array([0,1e4])
        y = np.array([0,1e4])
        y_p50 = y + perc50
        y_m50 = y - perc50
        y_p90 = y + perc90
        y_m90 = y - perc90

        ax.plot(x, y_p50, lw=0.5, dashes=[10,5], c='black',alpha=0.9)
        ax.plot(x, y_m50, lw=0.5, dashes=[10,5], c='black',alpha=0.9)
        ax.plot(x, y_p90, lw=0.5, ls='dotted', c='black',alpha=0.9)
        ax.plot(x, y_m90, lw=0.5, ls='dotted', c='black',alpha=0.9)
        ax.scatter(mag_nr, mag_surr, s=15, alpha=0.1, facecolor='C0',edgecolor="none")
        ax.scatter(mag_nr, mag_surr, s=15, alpha=0.5, facecolor='none',edgecolor='C0')

        cases = ["0021", "0283", "0353", "3144"]
        case_indices = [np.where(nr4500[:,0] == int(case)) for case in cases]
        highlight_nr = [mag_nr[case] for case in case_indices]
        highlight_surr = [mag_surr[case] for case in case_indices]
        ax.scatter(highlight_nr, highlight_surr, marker='x',s=25, alpha=1,color='C3')

        ax.set_xlim(0,10.5)
        ax.set_ylim(0,10.5)
        ax.set_xlabel("NR $v_k\;[0.001c]$")
        ax.set_ylabel("Surrogate $v_k\;[0.001c]$")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        bins = np.linspace(0, 10.5, 36)
        axt.hist(mag_nr, bins=bins, histtype='stepfilled',alpha=0.4,color='C0')
        axt.hist(mag_nr, bins=bins, histtype='step',color='C0')
        axt.set_xlim(0,10.5)
        axt.set_ylim(0,90)
        axt.axes.xaxis.set_ticklabels([])
        axt.xaxis.set_major_locator(MultipleLocator(2))
        axt.xaxis.set_minor_locator(MultipleLocator(0.5))
        axt.yaxis.set_major_locator(MultipleLocator(40))
        axt.yaxis.set_minor_locator(MultipleLocator(10))

        axr.hist(mag_surr, bins=bins, histtype='stepfilled', orientation='horizontal',alpha=0.4,color='C0')
        axr.hist(mag_surr, bins=bins, histtype='step', orientation='horizontal',color='C0')
        axr.set_xlim(0,90)
        axr.set_ylim(0,10.5)
        axr.axes.yaxis.set_ticklabels([])
        axr.xaxis.set_major_locator(MultipleLocator(40))
        axr.xaxis.set_minor_locator(MultipleLocator(10))
        axr.yaxis.set_major_locator(MultipleLocator(2))
        axr.yaxis.set_minor_locator(MultipleLocator(0.5))

        return fig

    @classmethod
    @plottingstuff
    def nr_comparison_profiles(self):
        '''Fig. 14. Comparison with SpEC simulations, time profiles.
        Usage: surrkick.plots.nr_comparison_profiles()'''

        w = 0.8
        h = 0.45
        gap = 0.1
        fig = plt.figure(figsize=(6, 6))
        ax_ll = fig.add_axes([0, 0, w, h])
        ax_lr = fig.add_axes([w + gap, 0, w, h])
        ax_ur = fig.add_axes([w + gap, h + gap, w, h])
        ax_ul = fig.add_axes([0, h + gap, w, h])

        # duplicated from histogram plot
        basename = os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/profile_case_id_"
        cases = ["0021", "0283", "0353", "3144"]
        filename = "nr_comparison_profiles.pkl"
        if not os.path.isfile(filename):
            surr_profiles = {}
            for case in cases:
                with open(basename+case+"/params.json", 'r') as f:
                    p = json.load(f)
                    q = p["relaxed-q"]
                    chi1 = p["surrogate-dimensionless-spin1"]
                    chi2 = p["surrogate-dimensionless-spin2"]
                    sk = surrkick(q=q, chi1=chi1, chi2=chi2, t_ref=-4500)
                    ts = sk.times
                    ps = - project(sk.Poft, sk.kickdir)
                    surr_profiles[case] = [ts, ps]
            print("Storing data:", filename)
            with open(filename, 'wb') as f: pickle.dump(surr_profiles, f)
        with open(filename, 'rb') as f: surr_data = pickle.load(f)

        axes = [ax_ll, ax_lr, ax_ur, ax_ul]
        for ax, case in zip(axes, cases):
            nr_data = np.loadtxt(basename+case+"/radiated_p.dat")
            tn = nr_data[:,0]
            pn = - nr_data[:,1] / 0.001
            ts = surr_data[case][0]
            ps = surr_data[case][1] / 0.001
            ax.plot(tn, pn, label = "NR",lw=2, color='C3',dashes=[8,4])
            ax.plot(ts, ps, label = "Surrogate",lw=2,color='C0')

            with open(basename+case+"/params.json", 'r') as f:
                p = json.load(f)
                sxs_id = str(p["SXS:BBH:ID"])
                q = p["relaxed-q"]
                q=min(q,1./q)
                chi1 = p["surrogate-dimensionless-spin1"]
                chi2 = p["surrogate-dimensionless-spin2"]

            label= "SXS:"+sxs_id+"\n"+'$q\simeq'+str(round(q,3) if round(q,3)!=1 else 1)+'$\n$\\boldsymbol{\\chi_1}\simeq'+str([round(x,3) if round(x,3)!=0 else 0 for x in chi1])+'$\n$\\boldsymbol{\\chi_2}\simeq'+str([round(x,3) if round(x,3)!=0 else 0 for x in chi2])+'$'
            ax.text(0.05, 0.95, label, verticalalignment='top',fontsize=14,transform=ax.transAxes)

            ax.set_xlim(-50, 50)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax_ul.legend(loc = "lower right")
        ax_ll.set_xlabel("$t\;\;[M]$")
        ax_lr.set_xlabel("$t\;\;[M]$")
        ax_ll.set_ylabel("$- \mathbf{P}(t) \cdot \mathbf{\hat v_k} \;\;[0.001 M]$")
        ax_ul.set_ylabel("$- \mathbf{P}(t) \cdot \mathbf{\hat v_k} \;\;[0.001 M]$")

        return fig

    @classmethod
    def timing(self):
        '''Surrkick code performance (no plot)
        Usage: surrkick.plots.timing()'''

        dim=1000
        surrogate().sur() # Load the surrogate once for all

        timessur=[]
        timeskick=[]
        timesall=[]
        for i in tqdm(range(dim)):

            q=np.random.uniform(0.5,1)
            phi = np.random.uniform(0,2*np.pi)
            theta = np.arccos(np.random.uniform(-1,1))
            r = 0.8*(np.random.uniform(0,1))**(1./3.)
            chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
            phi = np.random.uniform(0,2*np.pi)
            theta = np.arccos(np.random.uniform(-1,1))
            r = 0.8*(np.random.uniform(0,1))**(1./3.)
            chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
            sk= surrkick(q=q,chi1=chi1,chi2=chi2)

            t0=time.time()
            sk.hsample
            tsur=time.time()-t0

            t0=time.time()
            sk.kick
            tkick=time.time()-t0

            t0=time.time()
            sk=surrkick(q=q,chi1=chi1,chi2=chi2).kick
            tall=time.time()-t0

            timessur.append(tsur)
            timeskick.append(tkick)
            timesall.append(tall)

        print("Time, surrogate waveform:", np.mean(timessur),'s')
        print("Time, kick:", np.mean(timeskick),'s')
        print("Time, both:", np.mean(timesall),'s')

    @classmethod
    @plottingstuff
    def RIT_check(self):
        '''Fig. 15. Comparison with RIT simulations.
        Usage: surrkick.plots.RIT_check()'''

        # Read in tables cleaned from tex source
        kickdata=np.genfromtxt(os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/RITkicks.txt", dtype=(basestring,float,float,float),usecols=(0,9,11,13))
        simdata=np.genfromtxt(os.path.dirname(os.path.abspath(__file__))+"/"+"nr_comparison_data/RITdata.txt", dtype=(basestring,float,float,float,float,float),usecols=(0,6,7,8,9,10))

        parsed=[]
        filename='RIT_check.pkl'
        if not os.path.isfile(filename):
            print("Storing data:", filename)

            normRIT=[]
            kicksRIT=[]
            kicksmin=[]
            kicksmax=[]

            for kd,sd in tqdm(zip(kickdata,simdata)):

                simid, vx, vy, vz = kd
                simid2, sx, sy, sz, m1, m2 = sd
                # Sanity check. This is the same simulation...
                assert simid==simid2

                # Parse the confoiguration name
                # Extract the series name (e.g. "NTH15")
                series = simid.split('PH')[0]
                # Estraxt the configuration name (e.g. "N")
                configuration = simid.split('TH')[0].split('PH')[0]
                # Nominal mass ratio
                if "NQ" in configuration:
                    # IMPORTANT! Mass ratio here is m1/m2 in the RIT notation, so 2 and 0.5 are NOT the same thing.
                    qconf = float(configuration.split('NQ')[-1])/100.
                    configuration='NQ'
                else:
                    qconf=1. # All other configurations are nominally equal mass
                # Extract phi from configuration name
                phiconf = float(simid.split('PH')[-1])
                # Extract theta from configuration name
                if "TH" in simid:
                    thetaconf= float(simid.split('TH')[-1].split('PH')[0])
                else:
                    thetaconf=np.nan
                # Nominal spin values, converting RIT notation to our notation
                if configuration=='S':
                    chi1conf=0.8
                    chi2conf=0.8
                    theta1conf=thetaconf
                    theta2conf=180.-thetaconf
                    deltaphiconf=180.
                elif configuration=='K':
                    chi1mag=0.8
                    chi2mag=0.8
                    theta1conf=thetaconf
                    theta2conf=180.-thetaconf
                    deltaphiconf=0.
                elif configuration=='L':
                    chi1conf=0.8
                    chi2conf=0.8
                    theta1conf=0.
                    theta2conf=90.
                    deltaphiconf=0.
                elif configuration=='N':
                    chi1conf=0.
                    chi2conf=0.8
                    theta1conf=0 # Irrelevant
                    theta2conf=thetaconf
                    deltaphiconf=0 # Irrelevant
                elif configuration=='N9':
                    chi1conf=0.
                    chi2conf=0.9
                    theta1conf=0 # Irrelevant
                    theta2conf=thetaconf
                    deltaphiconf=0 # Irrelevant
                elif configuration=='NQ':
                    if qconf<1.: # Large BH is spinning
                        chi1conf=0.8
                        chi2conf=0.
                        theta1conf=thetaconf
                        theta2conf=0. # Irrelevant
                    else: # Small BH is spinning
                        qconf=1/qconf
                        chi1conf=0.
                        chi2conf=0.8
                        theta1conf=0. # Irrelevant
                        theta2conf=thetaconf
                    deltaphiconf=0. # Irrelevant

                # Extract kick magnitude
                kickRIT=convert.cisone(np.linalg.norm([vx,vy,vz]))

                # Actual spin values in the RIT simulations
                if configuration in ['N','N9','S','K','L']: # Spin in table is S1
                    chi1 = np.array([sx,sy,sz]) / m1**2
                    if configuration in ['N','N9']:
                        chi2=0*chi1
                    elif configuration=='S':
                        chi2=-chi1
                    elif configuration=='K':
                        chi2=np.array([chi1[0],chi1[1],-chi1[2]])
                    elif configuration=='L':
                        chi2=np.array([0,0,np.linalg.norm(chi1)])
                elif configuration=='NQ':
                    chi2 = np.array([sx,sy,sz]) / m2**2 # Spin in table is S1
                    chi1 = 0*chi2
                else:
                    raise ValueError("Configuration is", configuration)

                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=RuntimeWarning)
                    theta1,theta2,deltaphi,_=np.degrees(precession.build_angles([0,0,1],chi1,chi2))
                    if np.isnan(theta1):
                        theta1=0
                    if np.isnan(theta2):
                        theta2=0
                    if np.isnan(deltaphi):
                        deltaphi=0

                # In our notation, "1" is the larger BH. If that's not the case, flip the "1" and "2" labels
                if m1<m2:
                    m1,m2=m2,m1
                    chi1,chi2=chi2,chi1
                    theta1,theta2=theta2,theta1
                    deltaphi=-deltaphi
                q=m2/m1
                chi1mag = np.linalg.norm(chi1)
                chi2mag = np.linalg.norm(chi2)

                # Skip simulations which are outside the surrogate range (with some tolerance...)
                if q<0.49 or chi1mag>0.81 or chi2mag>0.81 :
                    continue

                # Check that values extracted in both ways more or less agree
                assert np.abs(min(qconf,1/qconf)-q)<2e-2
                if qconf==1: # In this case the labels don't matter
                    assert(np.abs(min(chi1conf,chi2conf)-min(chi1mag,chi2mag))<1e-2)
                    assert(np.abs(max(chi1conf,chi2conf)-max(chi1mag,chi2mag))<1e-2)
                    assert(np.abs(min(theta1conf,theta2conf)-min(theta1,theta2))<2e-1)
                    assert(np.abs(max(theta1conf,theta2conf)-max(theta1,theta2))<2e-1)
                else: # In this case labels matter
                    assert(np.abs(chi1conf-chi1mag)<1e-2)
                    assert(np.abs(chi2conf-chi2mag)<1e-2)
                    assert(np.abs(theta1conf-theta1)<2e-1)
                    assert(np.abs(theta2conf-theta2)<2e-1)

                theta1=np.radians(theta1)
                theta2=np.radians(theta2)
                deltaphi=np.radians(deltaphi)

                # Maximize kicks over t_ref using a maximization algorithm
                # Initial guesses
                nguess=1000
                t_vals = np.linspace(-4500,-100,int(nguess*2+1))[1:-1:2]
                guess=[surrkick(q=q,chi1=chi1,chi2=chi2,t_ref=t).kick for t in t_vals]
                # Find suitable bounds in alpha given the max/min array position
                def _boundsfromid(id):
                    if id==0: # Guess is at the beginning
                        return (-4500,t_vals[id+1])
                    elif id==len(t_vals)-1: # Guess is at the end
                        return (t_vals[id-1],-100)
                    else: # Guess is in the middle
                        return (t_vals[id-1],t_vals[id+1])
                # Find minimum, start from the smaller guess
                resmin = scipy.optimize.minimize_scalar(lambda t: surrkick(q=q,chi1=chi1,chi2=chi2,t_ref=t).kick , bounds=_boundsfromid(np.argmin(guess)),  method='bounded')
                kickmin = resmin.fun
                # Find maximum, start from the larger guess (note minus sign)
                resmax = scipy.optimize.minimize_scalar(lambda t: -surrkick(q=q,chi1=chi1,chi2=chi2,t_ref=t).kick , bounds=_boundsfromid(np.argmax(guess)),  method='bounded')
                kickmax = -resmax.fun

                # Normalize RIT kicks between 0 and 1
                normRIT.append((kickRIT - kickmin)/(kickmax-kickmin))
                kicksRIT.append(kickRIT)
                kicksmin.append(kickmin)
                kicksmax.append(kickmax)

            with open(filename, 'wb') as f: pickle.dump([normRIT,kicksRIT,kicksmin,kicksmax], f)

        with open(filename, 'rb') as f: normRIT,kicksRIT,kicksmin,kicksmax = np.array(pickle.load(f))

        print(convert.kms(max(kicksRIT-kicksmax)))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 0.8, 0.4])
        for x in [0,1]:
            ax.axvline(x,lw=3,c='black',alpha=0.7,ls='dashed')
        ax.hist(normRIT,bins=np.linspace(0,1.15,24),histtype='step',lw=3,alpha=1,color='C0')
        ax.hist(normRIT,bins=np.linspace(0,1.15,24),histtype='stepfilled',alpha=0.2,color='C0')
        ax.text(-0.15,11,"$\\textbf{"+str(len(normRIT[normRIT<0]))+"/"+str(len(normRIT))+"}$",horizontalalignment='center',transform=ax.transData,color='C3')
        ax.text(0.5,11,"$\\textbf{"+str(len(normRIT[np.logical_and(normRIT>0,normRIT<1)]))+"/"+str(len(normRIT))+"}$",horizontalalignment='center',transform=ax.transData,color='C2')
        ax.text(1.15,11,"$\\textbf{"+str(len(normRIT[normRIT>1]))+"/"+str(len(normRIT))+"}$",horizontalalignment='center',transform=ax.transData, color='C3')
        ax.set_xlim(-0.3,1.3)
        ax.set_ylim(0,13)

        ax.set_xlabel("$\\nu_k$")
        ax.set_ylabel("$N$")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 0.8, 0.8])

        return fig





########################################
if __name__ == "__main__":

    pass
    plots.minimal()

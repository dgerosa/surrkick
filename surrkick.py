''' Extract black hole kicks from gravitational-wave surrogate models. '''

from __future__ import print_function,division
import sys
import os
import time
import numpy as np
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import NRSur7dq2
#sur = NRSur7dq2.NRSurrogate7dq2('NRSur7dq2.h5')
from singleton_decorator import singleton
import precession

__author__ = "Davide Gerosa"
__email__ = "dgerosa@caltech.edu"
__license__ = "MIT"
__version__ = "dev"
__doc__="**Author** "+__author__+"\n\n"+\
        "**email** "+__email__+"\n\n"+\
        "**Licence** "+__license__+"\n\n"+\
        "**Version** "+__version__+"\n\n"+\
        __doc__


class summodes(object):
    '''Return indexes to perform a mode sum in l and m.'''

    @staticmethod
    def single(lmax):
        ''' Sum_{l,m} '''
        iterations=[]
        for l1 in np.arange(2,lmax+1):
            for m1 in np.arange(-l1,l1+1):
                    iterations.append((l1,m1))
        return iterations

    @staticmethod
    def double(lmax):
        ''' Sum_{l1,m1} Sum_{l2,m2}'''

        iterations=[]
        for l1 in np.arange(2,lmax+1):
            for m1 in np.arange(-l1,l1+1):
                for l2 in np.arange(2,lmax+1):
                    for m2 in np.arange(-l2,l2+1):
                        iterations.append((l1,m1,l2,m2))

        return iterations


class coeffs(object):
    ''' Coefficients of the linear momentum expression, from Eqs. (3.16-3.19) of arXiv:0707.4654. All are defined as static methods and can be called with, e.g., `coeffs.a(l,m)` withtout first defining an instance of the `coeffs` class.'''

    @staticmethod
    def a(l,m):
        '''Eq. (3.16) of arXiv:0707.4654'''
        return ( (l-m) * (l+m+1) )**0.5 / ( l * (l+1) )

    @staticmethod
    def b(l,m):
        '''Eq. (3.17) of arXiv:0707.4654'''
        return  ( 1/(2*l) ) *  ( ( (l-2) * (l+2) * (l+m) * (l+m-1) ) / ( (2*l-1) * (2*l+1) ))**0.5

    @staticmethod
    def c(l,m):
        '''Eq. (3.18) of arXiv:0707.4654'''
        return  2*m / ( l * (l+1) )

    @staticmethod
    def d(l,m):
        '''Eq. (3.19) of arXiv:0707.4654'''
        return  ( 1/l ) *  ( ( (l-2) * (l+2) * (l-m) * (l+m) ) / ( (2*l-1) * (2*l+1) ))**0.5

class convert(object):
    ''' Convert units to other units'''

    @staticmethod
    def kms(x):
        '''Convert a velocity from natural units c=1 to km/s. '''
        return x * 299792.458

@singleton
class surrogate(object):
    ''' Initialize the surrogate using a singleton. This means there can only be one instance of this class (makes sense: there's only one surrogate model).'''
    def __init__(self):
        self._sur=None

    def sur(self):
        if self._sur==None:
            self._sur = NRSur7dq2.NRSurrogate7dq2('NRSur7dq2.h5')
        return self._sur


class surrkick(object):
    '''
    Main class to compute kicks from surrogate models
    '''

    def __init__(self,q=1,chi1=[0,0,0],chi2=[0,0,0],times=None,spline_flag=False):
        '''
        Initialize the surrogate and a bunch of lazy-loading variables
        '''

        self.sur=surrogate().sur() # Initialize the surrogate. Note it's a singleton
        self.q = max(q,1/q) # Make sure q>1 in this class, that's what the surrogate wants
        self.chi1 = np.array(chi1)
        self.chi2 = np.array(chi2)
        # If times are specified use them, if not take the time nodes of the surrogate model
        if times==None:
            self.times = self.sur.t_coorb
        else:
            self.times = times

        self.spline_flag=spline_flag

        # Lazy loading, hidden variables
        self._hsample = None
        self._hint = None
        self._hdotsample = None
        self._hdotint = None
        self._hdotintfix = None
        self._hdotsamplefix = None
        self._lmax = None
        self._Eoftint=None
        self._Eoftsample=None
        self._Eoft=None
        self._Erad=None
        self._Poftint=None
        self._Poftsample=None
        self._Poft=None
        self._vkickcomp = None
        self._vkick = None
        self._vkickdir = None
        self._trajectorysample = None
        self._trajectoryint = None


    @property
    def hsample(self):
        '''Extract modes of strain h from the surrogate, evaluated at the surrogate time nodes.'''
        if self._hsample is None:
            self._hsample = self.sur(self.q, self.chi1, self.chi2,t=self.sur.t_coorb) # Returns a python dictionary with keys (l,m)
        return self._hsample

    @property
    def hint(self):
        '''Interpolate modes of h. This is the very same interpolation that is done inside the surrogate.'''
        if self._hint is None:
            # Split complex and real part... See `hdotintfix`
            self._hint = {k: [spline(self.sur.t_coorb,v.real),spline(self.sur.t_coorb,v.imag)] for k, v in self.hsample.items()}
        return self._hint

    @property
    def lmax(self):
        ''' Max l mode available in the surrogate model'''
        if self._lmax is None:
            self._lmax = sorted(self.hsample.keys())[-1][0]
        return self._lmax

    @property
    def hdotsample(self):
        '''Finite differencing derivative of the h modes, evaluated at the surrogate time nodes.'''
        if self._hdotsample is None:
            self._hdotsample = {k: np.gradient(v,edge_order=2)/np.gradient(self.times,edge_order=2) for k, v in self.hsample.items()}
        return self._hdotsample

    @property
    def hdotint(self):
        '''Interpolate dh/dt, first derivative of the strain. Return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used).'''
        if self._hdotint is None:
            self._hdotint = {k: [x.derivative() for x in v] for k, v in self.hint.items()}
        return self._hdotint

    def hdotsamplefix(self,l,m):
        '''Correct `hdotsample` to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used).'''
        if self._hdotsamplefix is None:
            if l<2 or l>self.lmax:
                return np.zeros(len(self.times),dtype=complex)
            elif m<-l or m>l:
                return np.zeros(len(self.times),dtype=complex)
            else:
                return self.hdotsample[l,m]

    def hdotintfix(self,l,m,t):
        '''Correct `hdot` to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used).'''
        if self._hdotintfix is None:

            if l<2 or l>self.lmax:
                return 0j
            elif m<-l or m>l:
                return 0j
            else:
                # Put complex and real part bach together... See `hint`
                return self.hdotint[l,m][0](t) + 1j*self.hdotint[l,m][1](t)

    @property
    def dEdtsample(self):
        '''Implement Eq. (3.8) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.'''

        dEdt = 0
        for l,m in summodes.single(self.lmax): # Use lmax+1 because sum involves terms with l-1

            # Eq. 3.8
            dEdt += (1/(16*np.pi)) * np.abs(self.hdotsamplefix(l,m))**2

        return dEdt

    def dEdtint(self,t):
        '''Implement Eq. (3.8) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.'''

        dEdt = 0
        for l,m in summodes.single(self.lmax): # Use lmax+1 because sum involves terms with l-1

            # Eq. 3.8
            dEdt += (1/(16*np.pi)) * np.abs(self.hdotintfix(l,m,t))**2

        return dEdt

    @property
    def Eoftsample(self):
        if self._Eoftsample is None:
            self._Eoftsample = np.array( [scipy.integrate.simps(self.dEdtsample[:i],self.times[:i]) for i in np.array(range(len(self.times)))+1])
        return self._Eoftsample

    @property
    def Eoftint(self):
        ''' Integrate the energy flux, to find the emitted energy as a function of time.'''
        if self._Eoftint is None:
            self._Eoftint = lambda t: scipy.integrate.odeint(lambda P,tx: self.dEdtint(tx), 0, np.append(self.times[0],t))[1:].flatten()
        return self._Eoftint

    def Eoft(self,t):
        ''' Evaluate E(t)'''

        try: len(t)
        except: t=[t]

        if not self.spline_flag and list(t)==list(self.times):
            return self.Eoftsample
        else:
            return self.Eoftint(t)

    @property
    def Erad(self):
        '''Component of the kick. Return vx vy vz; not 100% sure about the sign: is this the momentum emitted or the moentum of the recoil? Difference is just a minus sign somewhere.'''
        if self._Erad is None:

            if self.spline_flag is False:
                if self._Eoftsample is None: # Looks like you don't need the whole profile, but only the final kick
                    self._Erad= scipy.integrate.simps(self.dEdtsample,self.times)
                else:
                    self._Erad = self.Eoftsample[-1]
            else:
                self._Erad = self.Eoft(self.times[-1])

        return self._Erad




    @property
    def dPdtsample(self):
        '''Implement Eq. (3.14-3.15) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.'''

        dPpdt = 0
        dPzdt = 0
        for l,m in summodes.single(self.lmax+1): # Use lmax+1 because sum involves terms with l-1

            # Eq. 3.14. dPpdt= dPxdt + i dPydt
            dPpdt += (1/(8*np.pi)) * self.hdotsamplefix(l,m) * ( coeffs.a(l,m) * np.conj(self.hdotsamplefix(l,m+1)) + coeffs.b(l,-m) * np.conj(self.hdotsamplefix(l-1,m+1)) - coeffs.b(l+1,m+1) * np.conj(self.hdotsamplefix(l+1,m+1)) )
            # Eq. 3.15
            dPzdt += (1/(16*np.pi)) * self.hdotsamplefix(l,m) * ( coeffs.c(l,m) * np.conj(self.hdotsamplefix(l,m)) + coeffs.d(l,m) * np.conj(self.hdotsamplefix(l-1,m)) + coeffs.d(l+1,m) * np.conj(self.hdotsamplefix(l+1,m)) )

        dPxdt=dPpdt.real # From the definition of Pplus
        dPydt=dPpdt.imag # From the definition of Pplus
        dPzdt=dPzdt.real # Kill the imaginary part
        assert max(dPzdt.imag)<1e-6 # Check...

        return dPxdt,dPydt,dPzdt

    def dPdtint(self,t):
        '''Implement Eq. (3.14-3.15) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.'''

        dPpdt = 0
        dPzdt = 0
        for l,m in summodes.single(self.lmax+1): # Use lmax+1 because sum involves terms with l-1

            # Eq. 3.14. dPpdt= dPxdt + i dPydt
            dPpdt += (1/(8*np.pi)) * self.hdotintfix(l,m,t) * ( coeffs.a(l,m) * np.conj(self.hdotintfix(l,m+1,t)) + coeffs.b(l,-m) * np.conj(self.hdotintfix(l-1,m+1,t)) - coeffs.b(l+1,m+1) * np.conj(self.hdotintfix(l+1,m+1,t)) )
            # Eq. 3.15
            dPzdt += (1/(16*np.pi)) * self.hdotintfix(l,m,t) * ( coeffs.c(l,m) * np.conj(self.hdotintfix(l,m,t)) + coeffs.d(l,m) * np.conj(self.hdotintfix(l-1,m,t)) + coeffs.d(l+1,m) * np.conj(self.hdotintfix(l+1,m,t)) )

        dPxdt=dPpdt.real # From the definition of Pplus
        dPydt=dPpdt.imag # From the definition of Pplus
        dPzdt=dPzdt.real # Kill the imaginary part
        assert dPzdt.imag<1e-6 # Check...

        return dPxdt,dPydt,dPzdt

    @property
    def Poftsample(self):
        if self._Poftsample is None:
            self._Poftsample = np.array([ [scipy.integrate.simps(component[:i],self.times[:i]) for i in np.array(range(len(self.times)))+1] for component in self.dPdtsample])
        return self._Poftsample

    @property
    def Poftint(self):
        ''' Integrate the linear momentum flux, to find the linear momentum (or velocity since it's in mass units) as a function of time.'''
        if self._Poftint is None:
            self._Poftint = lambda t: np.transpose(scipy.integrate.odeint(lambda P,tx: self.dPdtint(tx),  [0,0,0], np.append(self.times[0],t))[1:])
        return self._Poftint

    def Poft(self,t):
        ''' Evaluate P(t)'''

        try: len(t)
        except: t=[t]

        if not self.spline_flag and list(t)==list(self.times):
            return self.Poftsample
        else:
            return self.Poftint(t)

    def voft(self,t):
        return self.Poft(t)

    @property
    def vkickcomp(self):
        '''Component of the kick. Return vx vy vz; not 100% sure about the sign: is this the momentum emitted or the moentum of the recoil? Difference is just a minus sign somewhere.'''
        if self._vkickcomp is None:

            if self.spline_flag is False:
                if self._Poftsample is None: # Looks like you don't need the whole profile, but only the final kick
                    self._vkickcomp= np.array([scipy.integrate.simps(component,self.times) for component in self.dPdtsample])
                else:
                    self._vkickcomp = np.array(self.Poftsample[:,-1])
            else:
                self._vkickcomp = self.Poft(self.times[-1])

        return self._vkickcomp

    @property
    def vkick(self):
        '''Magnitude of the kick'''
        if self._vkick is None:
            self._vkick = np.linalg.norm(self.vkickcomp)
        return self._vkick

    @property
    def vkickdir(self):
        if self._vkickdir is None:
            self._vkickdir = self.vkickcomp/self.vkick
        return self._vkickdir

    @property
    def trajectorysample(self):
        if self._trajectorysample is None:
            self._trajectorysample = np.array([ [scipy.integrate.simps(component[:i],self.times[:i]) for i in np.array(range(len(self.times)))+1] for component in self.Poftsample])

        return self._trajectorysample

    @property
    def trajectoryint(self):
        ''' Integrate the linear momentum flux, to find the linear momentum (or velocity since it's in mass units) as a function of time.'''
        if self._trajectoryint is None:
            raise NotImplementedError
            #print("here")
            #print(scipy.integrate.odeint(lambda P,tx: self.Poftint(tx).flatten(),  [0,0,0], np.append(self.times[0],10))[1:], Dfun= lambda P,tx: self.dPdtint(tx))
            #print("here")
            #self._trajectoryint = lambda t: np.transpose(scipy.integrate.odeint(lambda P,tx: self.Poftint(tx),  [0,0,0], np.append(self.times[0],t))[1:])
            #print(self._trajectoryint(10))
        return self._trajectoryint


    def trajectory(self,t):
        ''' Evaluate P(t)'''

        try: len(t)
        except: t=[t]

        if not self.spline_flag and list(t)==list(self.times):
            return self.trajectorysample
        else:
            return self.trajectoryint(t)



def project(timeseries,direction):
    ''' Project a 3D time series along some direction'''
    return np.array([np.dot(t,direction) for t in np.transpose(timeseries)])



def parseSXS(index):
    '''
    Get data from Chris' Moore parsing of the SXS catalog. Might be wrong, surely not accurate.

    Extract some info form the SXS data. By default only the total kick and its directio is extracted. If you want the fluxes as well, turn the flag on.
    TODO: Ideally this function should extract spin and mass ratio, initial directions etc etc from the SXS metadata
    '''
    root = '/Users/dgerosa/reps/kickedwaveform/RiccardoKicksProject/store.maths.cam.ac.uk/DAMTP/cjm96/kick_results/'
    trailindex="%04d" % (index,) # Trail zeros to the integer index
    folder= root+"SXS-"+trailindex

    # Extract metatada
    with open(folder+"/kicks.txt", "r") as openfile:
        for line in openfile:
            if "Radiated Linear Momentum X" in line: deltaPx = float(line.split()[5])
            if "Radiated Linear Momentum Y" in line: deltaPy = float(line.split()[5])
            if "Radiated Linear Momentum Z" in line: deltaPz = float(line.split()[5])
            if "Radiated Energy" in line and "Newtonian" not in line : Erad = float(line.split()[3])

    with open(folder+"/metadata"+trailindex+".txt") as openfile:
        for line in openfile:
            if "relaxed-mass1" in line: m1=float(line.split()[2])
            if "relaxed-mass2" in line: m2=float(line.split()[2])

    # Total kick speed
    kick = np.sqrt( deltaPx*deltaPx + deltaPy*deltaPy + deltaPz*deltaPz )
    q=min(m1/m2,m2/m1)
    return q, kick, Erad




class plots(object):
    ''' Do plots'''

    def plottingstuff(function):
        '''Use as decorator to handle plotting stuff'''

        #def wrapper(*args, **kw):
        def wrapper(self):

            # Before function call
            global plt
            from matplotlib import use #Useful when working on SSH
            use('Agg')
            from matplotlib import rc
            font = {'family':'serif','serif':['cmr10'],'weight' : 'medium','size' : 16}
            rc('font', **font)
            rc('text',usetex=True)
            rc('figure',max_open_warning=1000)
            rc('xtick',top=True)
            rc('ytick',right=True)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            #function(*args, **kw)
            function(self)

            # After function call
            plt.savefig(function.__name__+".pdf",bbox_inches='tight')
            plt.clf()

        return wrapper


    @classmethod
    @plottingstuff
    def nonspinning(self):

        fig = plt.figure(figsize=(6,6))
        L=0.7
        H=0.7
        S=0.2
        axall = [fig.add_axes([i*(S+H),0,L,H]) for i in [0,1]]


        q_vals = np.linspace(0.5,1,100)


        if True:

            vk_vals = []
            E_vals=[]
            t0=time.time()
            for q in q_vals:
                print(q)
                sk = surrkick(q=q , chi1=[0,0,0],chi2=[0,0,0],spline_flag=True)
                vk_vals.append(convert.kms(sk.vkick))
                E_vals.append(sk.Erad)

            print("Spline time", time.time()-t0)

            axall[0].plot(q_vals,vk_vals,label='surrogate (int)')
            axall[1].plot(q_vals,E_vals,label='surrogate (int)')

        if True:
            vk_vals = []
            E_vals=[]
            t0=time.time()
            for q in q_vals:
                print(q)
                sk = surrkick(q=q , chi1=[0,0,0],chi2=[0,0,0],spline_flag=False)
                vk_vals.append(convert.kms(sk.vkick))
                E_vals.append(sk.Erad)

            print("Samples time", time.time()-t0)

            axall[0].plot(q_vals,vk_vals,label='surrogate (samples)')
            axall[1].plot(q_vals,E_vals,label='surrogate (samples)')

        if True:
            fit=[convert.kms(precession.finalkick(0,0,0,q,0,0)) for q in q_vals]
            axall[0].plot(q_vals,fit,label='fit by Gonzalez 2007')
            #fit=[1-precession.finalmass(0,0,0,q,0,0) for q in q_vals]
            #axall[1].plot(q_vals,fit,label='fit by Barausse+ 2012')

        if True:
            # These are the sims with zero spins in the SXS public catalog
            qSXS,vkSXS,eradSXS=zip(*[parseSXS(i) for i in [74,70,72,73,1,66,87,86,67,71,68,69,91,90,2,180,198,8,100,93,7,194,184,169]])
            vkSXS=convert.kms(np.array(vkSXS))
            axall[0].scatter(qSXS,vkSXS,facecolor='none',edgecolor='red',label='SXS (?)')
            axall[1].scatter(qSXS,eradSXS,facecolor='none',edgecolor='red',label='SXS (?)')

        axall[0].set_xlabel("$q$")
        axall[1].set_xlabel("$q$")
        axall[0].set_ylabel("$v_k\;\;[\\rm km/s]$")
        axall[1].set_ylabel("$E_{\\rm rad}$")

        axall[0].legend(fontsize=15)
        axall[1].legend(fontsize=15)




        #plt.savefig('unequal.pdf',bbox_inches='tight')

    @classmethod
    @plottingstuff
    def profiles(self):

        fig = plt.figure(figsize=(6,6))
        L=0.7
        H=0.3
        S=0.05
        axall = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]

        q_vals = np.linspace(1,0.5,8)

        for i,q in enumerate(q_vals):
            print(q)
            color=plt.cm.copper(i/len(q_vals))
            b = surrkick(q=q , chi1=[0,0,0],chi2=[0,0,0],spline_flag=False)
            axall[0].plot(b.times,b.voft(b.times)[0]*1000,color=color,alpha=0.7)
            axall[1].plot(b.times,b.voft(b.times)[1]*1000,color=color,alpha=0.7)
            axall[2].plot(b.times,b.voft(b.times)[2]*1000,color=color,alpha=0.7)
            axall[3].plot(b.times,project(b.voft(b.times),b.vkickdir)*1000,color=color,alpha=0.7)

        for ax in axall:
            ax.set_xlim(-50,50)
            ax.set_ylim(-1,1)
        for ax in axall[:-1]:
            ax.set_xticklabels([])
        axall[-1].set_xlabel("$t\;\;[M]$")
        for ax,d in zip(axall,["x","y","z","v_k"]):
            ax.set_ylabel("$\mathbf{v}(t)\cdot \hat\mathbf{"+d+"} \;\;[0.001c]$")

    @classmethod
    @plottingstuff
    def centerofmass(self):

        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

        q=0.8
        sk = surrkick(q=q , chi1=[0,0,0],chi2=[0,0,0])

        x0,y0,z0=np.transpose(sk.trajectory(sk.times))[sk.times==min(abs(sk.times))][0]
        x,y,z=np.transpose(np.transpose(sk.trajectory(sk.times))[np.logical_and(sk.times>-1000,sk.times<20)])
        ax.plot(x-x0,y-y0,z-z0)

        ax.scatter(0,0,0,marker='.',s=40,alpha=0.5)
        x,y,z=sk.trajectory(sk.times)
        vx,vy,vz=sk.voft(sk.times)
        for t in [-50,-10,-2,2,10,18]:
            i = np.abs(sk.times - t).argmin()
            v=np.linalg.norm([vx[i],vy[i],vz[i]])
            arrowsize=1e-4
            ax.quiver(x[i]-x0,y[i]-y0,z[i]-z0,vx[i]*arrowsize/v,vy[i]*arrowsize/v,vz[i]*arrowsize/v,length=0.001,arrow_length_ratio=500,alpha=0.5)

        ax.set_xlim(-0.001,0.001)
        ax.set_ylim(-0.001,0.001)
        ax.set_zlim(-0.0001,0.0003)


########################################
if __name__ == "__main__":
    #
    sk=surrkick()
    #print(sk.Eoftsample)
    #print(sk.Eoft([10,20]))


    # print(sk.vkick)
    # sk=surrkick(spline_flag=False)
    # print(sk.vkick)
    #
    #sk=surrkick(q=0.8,spline_flag=False)
    #print(sk.trajectory(sk.times))


    #print(sk.trajectory(sk.times))
    #print(sk.Poft(np.array([10,29])))

    plots.nonspinning()
    #plots.profiles()
    #plots.centerofmass()
    #
    # sur=surrogate()
    # print(sur)
    # sur2=surrogate()
    # print(sur2)

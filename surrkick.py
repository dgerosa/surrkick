''' Extract black hole kicks from gravitational-wave surrogate models. '''

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
import precession
import h5py
from tqdm import tqdm
import cPickle as pickle


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
    ''' Coefficients of the momentum expression, from Eqs. (3.16-3.19,3.25) of arXiv:0707.4654. All are defined as static methods and can be called with, e.g., `coeffs.a(l,m)` withtout first defining an instance of the `coeffs` class.'''

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

    @staticmethod
    def f(l,m):
        '''Eq. (3.25) of arXiv:0707.4654'''
        return  ( l*(l+1) - m*(m+1) )**0.5



class convert(object):
    ''' Convert units to other units'''

    @staticmethod
    def kms(x):
        '''Convert a velocity from natural units c=1 to km/s. '''
        return x * 299792.458

    @staticmethod
    def anglestocoords(chi1mag,chi2mag,theta1,theta2,deltaphi,phi1):

        assert theta1>=0 and theta1<=np.pi and theta2>=0 and theta2<=np.pi
        phi2 = deltaphi+phi1
        chi1 = chi1mag*np.array([np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)])
        chi2 = chi2mag*np.array([np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2)])
        return chi1,chi2

    @staticmethod
    def coordstoangles(chi1,chi2):

        chi1mag = np.linalg.norm(chi1)
        chi2mag = np.linalg.norm(chi2)

        theta1 = np.arccos(chi1[2]/chi1mag)
        theta2 = np.arccos(chi2[2]/chi2mag)
        #print(theta1,chi1[0],chi1mag,chi1[0]/(chi1mag*np.sin(theta1)))
        if chi1[1]==0:
            phi1 = np.sign(chi1[0])*np.arccos(chi1[0]/(chi1mag*np.sin(theta1)))
        else:
            phi1 = np.sign(chi1[1])*np.arccos(chi1[0]/(chi1mag*np.sin(theta1)))
        if chi2[1]==0:
            phi2 = np.sign(chi2[0])*np.arccos(chi2[0]/(chi2mag*np.sin(theta2)))
        else:
            phi2 = np.sign(chi2[1])*np.arccos(chi2[0]/(chi2mag*np.sin(theta2)))
        deltaphi = phi2 - phi1

        return chi1mag,chi2mag,theta1,theta2,deltaphi,phi1


@singleton
class surrogate(object):
    ''' Initialize the surrogate using a singleton. This means there can only be one instance of this class (makes sense: there's only one surrogate model).'''
    def __init__(self):
        self._sur=None

    def sur(self):
        if self._sur==None:
            self._sur = NRSur7dq2.NRSurrogate7dq2('NRSur7dq2.h5')
        return self._sur


class bhbin(object):
    ''' Super class to contain info for a single binary evolution'''

    def __init__(self,q=1,chi1=[0,0,0],chi2=[0,0,0],t_ref=-100):
        self.sur=surrogate().sur() # Initialize the surrogate. Note it's a singleton
        self.q = max(q,1/q) # Make sure q>1 in this class, that's what the surrogate wants
        self.chi1 = np.array(chi1) # chi1 is the spin of the larger BH
        self.chi2 = np.array(chi2) # chi2 is the spin of the smaller BH
        self.times = self.sur.t_coorb # Short name for the time nodes

        assert t_ref>=-4500 and t_ref<=-100
        if t_ref==-4500:
            self.t_ref = None
        else:
            self.t_ref=t_ref

        self._hsample = None
        self._hdotsample = None
        self._lmax = None
        self._dEdt = None
        self._Eoft = None
        self._Erad = None
        self._dPdt = None
        self._Poft = None
        self._Prad = None
        self._dJdt = None
        self._Joft = None
        self._Jrad = None
        self._xoft = None

    @property
    def hsample(self):
        '''Extract modes of strain h from the surrogate, evaluated at the surrogate time nodes.'''
        if self._hsample is None:
            self._hsample = self.sur(self.q, self.chi1, self.chi2,t=self.times,t_ref=self.t_ref) # Returns a python dictionary with keys (l,m)
        return self._hsample

    @property
    def lmax(self):
        ''' Max l mode available in the surrogate model'''
        if self._lmax is None:
            self._lmax = sorted(self.hsample.keys())[-1][0]
        return self._lmax

    def h(self,l,m):
        '''Correct the strain values to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used).'''
        if l<2 or l>self.lmax:
            return np.zeros(len(self.times),dtype=complex)
        elif m<-l or m>l:
            return np.zeros(len(self.times),dtype=complex)
        else:
            return self.hsample[l,m]

    @property
    def hdotsample(self):
        '''Derivative of the strain at the time nodes. First interpolate the h with a spline, derivate the spline and evaluate that derivative at the nodes.'''
        if self._hdotsample is None:

            self._hdotsample =  {k: spline(self.times,v.real).derivative()(self.times)+1j*spline(self.times,v.imag).derivative()(self.times) for k, v in self.hsample.items()}

        return self._hdotsample

    def hdot(self,l,m):
        '''Correct the strain derivative values to return zero if either l or m are not allowed (this is how the expressions of arXiv:0707.4654 are supposed to be used).'''
        if l<2 or l>self.lmax:
            return np.zeros(len(self.times),dtype=complex)
        elif m<-l or m>l:
            return np.zeros(len(self.times),dtype=complex)
        else:
            return self.hdotsample[l,m]


    @property
    def dEdt(self):
        '''Implement Eq. (3.8) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there.'''

        if self._dEdt is None:
            dEdt = 0
            for l,m in summodes.single(self.lmax):

                # Eq. 3.8
                dEdt += (1/(16*np.pi)) * np.abs(self.hdot(l,m))**2

            self._dEdt = dEdt

        return self._dEdt

    @property
    def Eoft(self):
        if self._Eoft is None:
            # The integral of a spline is called antiderivative (mmmh...)

            origin=0
            self._Eoft = spline(self.times,self.dEdt).antiderivative()(self.times)-origin
        return self._Eoft

    @property
    def Erad(self):
        ''' Total energy momentum radiated'''
        if self._Erad is None:
            self._Erad = self.Eoft[-1]
        return self._Erad

    @property
    def dPdt(self):
        '''Implement Eq. (3.14-3.15) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there. '''

        if self._dPdt is None:

            dPpdt = 0
            dPzdt = 0

            for l,m in summodes.single(self.lmax):

                # Eq. 3.14. dPpdt= dPxdt + i dPydt
                dPpdt += (1/(8*np.pi)) * self.hdot(l,m) * ( coeffs.a(l,m) * np.conj(self.hdot(l,m+1)) + coeffs.b(l,-m) * np.conj(self.hdot(l-1,m+1)) - coeffs.b(l+1,m+1) * np.conj(self.hdot(l+1,m+1)) )
                # Eq. 3.15
                dPzdt += (1/(16*np.pi)) * self.hdot(l,m) * ( coeffs.c(l,m) * np.conj(self.hdot(l,m)) + coeffs.d(l,m) * np.conj(self.hdot(l-1,m)) + coeffs.d(l+1,m) * np.conj(self.hdot(l+1,m)) )

            dPxdt=dPpdt.real # From the definition of Pplus
            dPydt=dPpdt.imag # From the definition of Pplus
            assert max(dPzdt.imag)<1e-6 # Check...
            dPzdt=dPzdt.real # Kill the imaginary part


            self._dPdt = np.transpose([dPxdt,dPydt,dPzdt])

        return self._dPdt

    @property
    def Poft(self):
        if self._Poft is None:
            # The integral of a spline is called antiderivative (mmmh...)

            #print("here")
            #origin = [np.average(x) for x in np.transpose(self.dvdt[0:100])]
            origin=[0,0,0]
            #origin = np.array([spline(self.times,v).antiderivative()(self.times[30])  for v in np.transpose(self.dvdt)])
            self._Poft = np.transpose([spline(self.times,v).antiderivative()(self.times)-o  for v,o in zip(np.transpose(self.dPdt),origin)])
            #print(origin)
            #print(self._v[-1])
            #print("")
            #sys.exit()
        return self._Poft

    @property
    def Prad(self):
        ''' Total linear momentum radiated'''
        if self._Prad is None:
            self._Prad = np.linalg.norm(self.Poft[-1])
        return self._Prad

    @property
    def voft(self):
        ''' Velocity of the remnant is minus the emitted momentum'''
        return -self.Poft

    @property
    def kick(self):
        '''Final kick velocity'''
        return self.Prad

    @property
    def kickcomp(self):
        '''Components of the kick in the same xyz plane of the surrogate'''
        return self.voft[-1]

    @property
    def kickdir(self):
        '''Kick direction in the same xyz plane of the surrogate'''
        return self.kickcomp/self.kick


    @property
    def dJdt(self):
        '''Implement Eq. (3.22-3.24) of arXiv:0707.4654 for the linear momentum flux. Note that the modes provided by the surrogate models are actually h*(r/M) extracted as r=infinity, so the r^2 factor is already in there. '''

        if self._dJdt is None:

            dJxdt = 0
            dJydt = 0
            dJzdt = 0

            for l,m in summodes.single(self.lmax):

                # Eq. 3.22
                dJxdt += (1/(32*np.pi)) * self.h(l,m) * ( coeffs.f(l,m) * np.conj(self.hdot(l,m+1)) + coeffs.f(l,-m) * np.conj(self.hdot(l,m-1)) )
                # Eq. 3.23
                dJydt += (-1/(32*np.pi)) * self.h(l,m) * ( coeffs.f(l,m) * np.conj(self.hdot(l,m+1)) - coeffs.f(l,-m) * np.conj(self.hdot(l,m-1)) )
                # Eq. 3.24
                dJzdt += (1/(16*np.pi)) * m * self.h(l,m) * np.conj(self.hdot(l,m))

            dJxdt=dJxdt.imag
            dJydt=dJydt.real
            dJzdt=dJzdt.imag


            self._dJdt = np.transpose([dJxdt,dJydt,dJzdt])

        return self._dJdt

    @property
    def Joft(self):
        if self._Joft is None:

            # The integral of a spline is called antiderivative (mmmh...)
            origin=[0,0,0]
            self._Joft = np.transpose([spline(self.times,v).antiderivative()(self.times)-o  for v,o in zip(np.transpose(self.dJdt),origin)])

        return self._Joft

    @property
    def Jrad(self):
        ''' Total linear momentum radiated'''
        if self._Jrad is None:
            self._Jrad = np.linalg.norm(self.Joft[-1])
        return self._Jrad

    @property
    def xoft(self):
        if self._xoft is None:
            # The integral of a spline is called antiderivative (mmmh...)

            #print("here")
            #origin = [np.average(x) for x in np.transpose(self.dvdt[0:100])]
            origin=[0,0,0]
            #origin = np.array([spline(self.times,v).antiderivative()(self.times[30])  for v in np.transpose(self.dvdt)])
            self._xoft = np.transpose([spline(self.times,v).antiderivative()(self.times)-o  for v,o in zip(np.transpose(self.voft),origin)])
            #print(origin)
            #print(self._v[-1])
            #print("")
            #sys.exit()
        return self._xoft



def project(timeseries,direction):
    ''' Project a 3D time series along some direction'''
    return np.array([np.dot(t,direction) for t in timeseries])


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
            if "Radiated Angular Momentum X" in line: deltaJx = float(line.split()[5])
            if "Radiated Angular Momentum Y" in line: deltaJy = float(line.split()[5])
            if "Radiated Angular Momentum Z" in line: deltaJz = float(line.split()[5])

    with open(folder+"/metadata"+trailindex+".txt") as openfile:
        for line in openfile:
            if "relaxed-mass1" in line: m1=float(line.split()[2])
            if "relaxed-mass2" in line: m2=float(line.split()[2])

    Prad = np.sqrt( deltaPx*deltaPx + deltaPy*deltaPy + deltaPz*deltaPz )
    Jrad = np.sqrt( deltaJx*deltaJx + deltaJy*deltaJy + deltaJz*deltaJz )
    q=min(m1/m2,m2/m1)

    return q, [Erad, Prad, Jrad]


def fitkick(q,chi1m,chi2m,theta1,theta2,deltaphi,bigTheta):

    '''
    Estimate the final kick of the BH remnant following a BH merger. We
    implement the fitting formula to numerical relativity simulations developed
    by the Rochester group. The implementation is the same as reported in Gerosa
    and Kesden 2016
    '''

    q = min(q,1/q)  #Make sure q<1 in here
    # Spins here are defined in a frame with L along z and S1 in xz
    eta=q*pow(1.+q,-2.)
    hatL = np.array([0,0,1])
    hatS1 = np.array([np.sin(theta1),0,np.cos(theta1)])
    hatS2 = np.array([np.sin(theta2)*np.cos(deltaphi),np.sin(theta2)*np.sin(deltaphi),np.cos(theta2)])
    #Useful spin combinations.
    Delta = (q*chi2m*hatS2-chi1m*hatS1)/(1.+q)
    Delta_par = np.dot(Delta,hatL)
    Delta_perp = np.linalg.norm(np.cross(Delta,hatL))
    chit = (q*q*chi2m*hatS2+chi1m*hatS1)/pow(1.+q,2.)
    chit_par = np.dot(chit,hatL)
    chit_perp = np.linalg.norm(np.cross(chit,hatL))

    #Kick. Coefficients are quoted in km/s

    # vm and vperp are like in Kesden at 2010a, vpar is modified from Lousto Zlochower 2013
    zeta=np.radians(145.)
    A=1.2e4
    B=-0.93
    H=6.9e3

    # Switch on/off the various (super)kick contribution. Default are all on
    superkick=True
    hangupkick=True
    crosskick=True

    if superkick==True:
        V11=3677.76
    else:
        V11=0.
    if hangupkick==True:
        VA=2481.21
        VB=1792.45
        VC=1506.52
    else:
        VA=0.
        VB=0.
        VC=0.
    if crosskick==True:
        C2=1140.
        C3=2481.
    else:
        C2=0.
        C3=0.

    vm=A*eta*eta*(1.+B*eta)*(1.-q)/(1.+q)
    vperp=H*eta*eta*Delta_par
    vpar=16.*eta*eta* (Delta_perp*(V11+2.*VA*chit_par+4.*VB*pow(chit_par,2.)+8.*VC*pow(chit_par,3.)) + chit_perp*Delta_par*(2.*C2+4.*C3*chit_par)) * np.cos(bigTheta)
    vkick=np.linalg.norm([vm+vperp*np.cos(zeta),vperp*np.sin(zeta),vpar])

    #print(vkick)

    assert vkick<5000 #I got v_kick>5000km/s. This shouldn't be possibile"
    vkick=vkick/299792.458  # Natural units


    return vkick


class optkick(object):

    def __init__(self,q=1,chi1=[0,0,0],chi2=[0,0,0],t_ref=None):
        self.q = q # Mass ratio
        self.chi1 = np.array(chi1) # chi1 is the spin of the larger BH
        self.chi2 = np.array(chi2) # chi2 is the spin of the smaller BH
        self.t_ref=t_ref
        self.chi1m,self.chi2m,self.theta1,self.theta2,self.deltaphi,dummy = convert.coordstoangles(self.chi1,self.chi2)


    def _phasefit(self,x):
        return fitkick(self.q,self.chi1m,self.chi2m,self.theta1,self.theta2,self.deltaphi,x)
    def phasefit(self,xs):
        return np.array([self._phasefit(x) for x in xs])

    def _phasesur(self,x):
        chi1h,chi2h=convert.anglestocoords(self.chi1m,self.chi2m,self.theta1,self.theta2,self.deltaphi,x)
        return bhbin(q=self.q,chi1=chi1h,chi2=chi2h,t_ref=self.t_ref).kick
    def phasesur(self,xs):
        return np.array([self._phasesur(x) for x in xs])

    def find(self,flag):
        if flag=='fit':
            phasefunc=self._phasefit
        elif flag=='sur':
            phasefunc=self._phasesur
        else:
            raise InputError, "set flag equal to 'fit' or 'sur'"


        xs=np.linspace(-np.pi,np.pi,300)
        evals= np.array([phasefunc(x) for x in xs])
        minx = xs[evals==min(evals)][0]
        maxx = xs[evals==max(evals)][0]

        #print(minx,maxx)

        phasemin=scipy.fmin(lambda x:phasefunc(x),minx)
        phasemax=scipy.fmin(lambda x:-phasefunc(x),maxx)

        #phasemin,phasemax = [scipy.optimize.fminbound(lambda x: sign*phasefunc(x),-np.pi,np.pi) for sign in [1,-1]]
        kickmin,kickmax = [phasefunc(x) for x in phasemin,phasemax]

        return kickmin,kickmax



class plots(object):
    ''' Do plots'''

    def plottingstuff(function):
        '''Use as decorator to handle plotting stuff'''

        #def wrapper(*args, **kw):
        def wrapper(self):

            # Before function call
            global plt,AutoMinorLocator,MultipleLocator
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
            from matplotlib.ticker import AutoMinorLocator,MultipleLocator
            pp= PdfPages(function.__name__+".pdf")

            #function(*args, **kw)
            fig = function(self)

            try:
                len(fig)
            except:
                fig=[fig]
            for f in tqdm(fig):
                f.savefig(pp, format='pdf',bbox_inches='tight')
                f.clf()
            pp.close()
        return wrapper


    @classmethod
    @plottingstuff
    def nonspinning(self):



        figP = plt.figure(figsize=(6,6))
        L=0.7
        H=0.7
        S=0.2
        axP = [figP.add_axes([i*(S+H),0,L,H]) for i in [0,1,2]]


        q_vals = np.linspace(0.5,1,100)

        if True:
            data=[]
            t0=time.time()
            for q in q_vals:
                print(q)
                sk = bhbin(q=q)
                data.append([sk.Erad,sk.Prad,sk.Jrad])

            print("Time", time.time()-t0)

            for ax, d in zip(axall,zip(*data)):
                ax.plot(q_vals,d,label='surrogate (samples)')

        if False:
            fit=[precession.finalkick(0,0,0,q,0,0) for q in q_vals]
            axall[1].plot(q_vals,fit,label='fit by Gonzalez 2007')

        if True:
            # These are the sims with zero spins in the SXS public catalog
            q_vals, data = zip(*[parseSXS(i) for i in [74,70,72,73,1,66,87,86,67,71,68,69,91,90,2,180,198,8,100,93,7,194,184,169]])


            for ax, d in zip(axall,zip(*data)):
                ax.scatter(q_vals,d,facecolor='none',edgecolor='red',label='surrogate (int)')

        #axall[0].set_xlabel("$q$")
        #axall[1].set_xlabel("$q$")
        #axall[0].set_ylabel("$v_k\;\;[\\rm km/s]$")
        #axall[1].set_ylabel("$E_{\\rm rad}$")

        #axall[0].legend(fontsize=15)
        #axall[1].legend(fontsize=15)

        return fig

    @classmethod
    @plottingstuff
    def nospinprofiles(self):

        L=0.7
        H=0.3
        S=0.05


        figP = plt.figure(figsize=(6,6))
        axP = [figP.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
        figE = plt.figure(figsize=(6,6))
        axE = figE.add_axes([0,0,L,H])
        figJ = plt.figure(figsize=(6,6))
        axJ = [figJ.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]

        q_vals = np.linspace(1,0.5,8)

        for i,q in enumerate(q_vals):
            print(q)
            color=plt.cm.copper(i/len(q_vals))
            b = bhbin(q=q)
            #if q==q_vals[-1]:
            #    dashes=[10,2]
            #else:
            #    dashes=''
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
        for ax,d in zip(axP,["x","y","z","v_k"]):
            ax.set_ylim(-1,1)
            ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")
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


        allfig=[]


        if True:
            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=0.5
            chi1=[0,0,0]
            chi2=[0,0,0]
            sk = bhbin(q=q , chi1=chi1,chi2=chi2)
            print(sk.times)
            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]

            x,y,z=np.transpose(sk.xoft[np.logical_and(sk.times>-4000,sk.times<26)])
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

            #ax.set_title('$q='+str(q)+'\;\;\chi_1='+str(chi1)+'\;\;\chi_2='+str(chi2)+'$')
            ax.set_xticklabels(ax.get_xticks(), fontsize=11)
            ax.set_yticklabels(ax.get_yticks(), fontsize=11)
            ax.set_zticklabels(ax.get_zticks(), fontsize=11)
            fig.text(0.38,0.45,'$q='+str(q)+'$\n$\\chi_1=0$\n${\\chi_2}=0$',transform=fig.transFigure)

            ax.set_xlabel("$x\;\;[M]$")
            ax.set_ylabel("$y\;\;[M]$")
            ax.set_zlabel("$z\;\;[M]$")
            ax.xaxis.labelpad=6
            ax.yaxis.labelpad=8
            ax.zaxis.labelpad=4

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.zaxis.set_minor_locator(AutoMinorLocator())

            allfig.append(fig)


        if True:
            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=0.5
            chi1=[0.8,0,0]
            chi2=[-0.8,0,0]
            sk = bhbin(q=q , chi1=chi1, chi2=chi2)

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

            ax.set_xticklabels(ax.get_xticks(), fontsize=11)
            ax.set_yticklabels(ax.get_yticks(), fontsize=11)
            ax.set_zticklabels(ax.get_zticks(), fontsize=11)
            fig.text(0.1,0.4,'$q='+str(q)+'$\n$\\boldsymbol{\\chi_1}=[0.8,0,0]$\n$\\boldsymbol{\\chi_2}=[-0.8,0,0]$',transform=fig.transFigure)

            ax.set_xlabel("$x\;\;[M]$")
            ax.set_ylabel("$y\;\;[M]$")
            ax.set_zlabel("$z\;\;[M]$")
            ax.xaxis.labelpad=6
            ax.yaxis.labelpad=8
            ax.zaxis.labelpad=4

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.zaxis.set_minor_locator(AutoMinorLocator())

            allfig.append(fig)




        if False:
            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7], projection='3d')

            q=0.8
            chi1=[0.8,0,0]
            chi2=[-0.8,0,0]
            sk = bhbin(q=q , chi1=chi1, chi2=chi2)

            x0,y0,z0=sk.xoft[sk.times==min(abs(sk.times))][0]

            x,y,z=np.transpose(sk.xoft[np.logical_and(sk.times>-1000,sk.times<19)])
            ax.plot(x-x0,y-y0,z-z0)

            ax.scatter(0,0,0,marker='.',s=40,alpha=0.5)
            x,y,z=np.transpose(sk.xoft)
            vx,vy,vz=np.transpose(sk.voft)
            for t in [-900,-50,-10,-5,-2,2,5,6,18]:
                i = np.abs(sk.times - t).argmin()
                v=np.linalg.norm([vx[i],vy[i],vz[i]])
                arrowsize=1e-4
                ax.quiver(x[i]-x0,y[i]-y0,z[i]-z0,vx[i]*arrowsize/v,vy[i]*arrowsize/v,vz[i]*arrowsize/v,length=0.0001,arrow_length_ratio=9000,alpha=0.5)

            ax.set_xlim(-0.0008,0.0008)
            ax.set_ylim(-0.0008,0.0008)
            ax.set_zlim(-0.0008,0.0008)

            ax.set_title('$q='+str(q)+'\;\;\chi_1='+str(chi1)+'\;\;\chi_2='+str(chi2)+'$')
            ax.set_xticklabels(ax.get_xticks(), fontsize=13)
            ax.set_yticklabels(ax.get_yticks(), fontsize=13)
            ax.set_zticklabels(ax.get_zticks(), fontsize=13)


            allfig.append(fig)


        return allfig


    @classmethod
    @plottingstuff
    def alphaseries(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        fig = plt.figure(figsize=(6,6))
        L=0.7
        H=0.3
        S=0.05
        axs = [fig.add_axes([i*(S+L),0,L,H]) for i in [0,1]]

        dim=100

        #chi_vals= np.linspace(0,0.8,9)
        chimag=0.8

        tref_vals=np.linspace(-250,-100,dim)
        kick_vals=[]

        for t_ref in tqdm(tref_vals):
            sk = bhbin(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=t_ref)
            kick_vals.append(sk.kick)
            #print(t_ref,sk.kick)
        axs[0].plot(tref_vals,1/0.001*np.array(kick_vals))
        axs[0].scatter(-125,1/0.001*bhbin(q=1 , chi1=[chimag,0,0],chi2=[-chimag,0,0],t_ref=-125).kick,marker='o',edgecolor='C1',facecolor='none',s=100,linewidth='2')


        alpha_vals=np.linspace(-np.pi,np.pi,dim)

        kick_vals=[]
        for alpha in tqdm(alpha_vals):
            sk = bhbin(q=1 , chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])
            kick_vals.append(sk.kick)

        axs[1].plot(alpha_vals,1/0.001*np.array(kick_vals),c='C3')

        axs[0].text(0.05,0.5,'$q=1$\n$\chi_1=\chi_2=0.8$\n$\\alpha=0$',transform=axs[0].transAxes,linespacing=1.4)
        axs[1].text(0.05,0.5,'$q=1$\n$\chi_1=\chi_2=0.8$\n$t_{\\rm ref}=-100M$',transform=axs[1].transAxes,linespacing=1.4)


        axs[1].set_yticklabels([])
        #     #chimag=0.723
        #     alpha_vals=np.linspace(0,2.*np.pi,50)
        #     kick_vals=[]
        #     for alpha in alpha_vals:
        #         sk = bhbin(q=1 , chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])
        #         kick_vals.append(convert.kms(sk.kickcomp[-1]))
        #
        #     ax.plot(alpha_vals,kick_vals,color=color)
        #     #ax.plot(alpha_vals,2725*np.cos(0.98*np.pi+alpha_vals))
        #
        #

        axs[0].set_xlabel("$t_{\\rm ref}\;\;[M]$")


        axs[1].set_xlim(-1.1*np.pi,1.1*np.pi)
        axs[1].set_xlabel("$\\alpha$")
        axs[1].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        axs[1].set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])


        axs[0].set_ylabel("${v_k} \;\;[0.001 c]$")
        #
        for ax in axs:
            ax.set_ylim(0,10)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        return fig

    @classmethod
    @plottingstuff
    def alphaprof(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.3])
        chimag=0.5

        ax.axvline(0,c='black',alpha=0.3,ls='dotted')
        ax.axhline(0,c='black',alpha=0.3,ls='dotted')

        alpha_vals=np.linspace(-np.pi,np.pi,50)
        kick_vals=[]
        for i, alpha in tqdm(enumerate(alpha_vals)):
            sk = bhbin(q=1, chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])

            color=plt.cm.copper(i/len(alpha_vals))

            #kick_vals.append(convert.kms(sk.kickcomp[-1]))

            ax.plot(sk.times,1/0.001*project(sk.voft,sk.kickdir),color=color,alpha=0.8)
            #ax.plot(alpha_vals,2725*np.cos(0.98*np.pi+alpha_vals))


        ax.set_xlim(-100,50)
        #ax.set_ylim(-3000,3000)
        ax.set_xlabel("$t\;\;[M]$")
        ax.set_ylabel("$\mathbf{v}(t)\cdot \mathbf{\hat v_k} \;\;[0.001 c]$")
        #ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
        #ax.set_xticklabels(['$0$','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.text(0.05,0.55,'$q=1$\n$\chi_1=\chi_2=0.8$\n$\\alpha=-\pi ... \pi$',transform=ax.transAxes,linespacing=1.4)

        return fig

    @classmethod
    @plottingstuff
    def lineofsight(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        figs=[]
        L=0.7
        H=0.6
        S=0.05



        for q in [0.5]:

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_axes([0,0,L,H])
            ax.axhline(0,c='black',alpha=0.3,ls='dotted')

            chimag=0.8
            sk= bhbin(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag],t_ref=-100)

            store=[]
            dim=15
            for i in tqdm(np.linspace(0,1,dim)):


                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                randomvec= [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ]
                #print(randomvec)
                store.append([randomvec,project(sk.voft,randomvec)[-1]])

            #for x in store:
            #    print(x)

            store=sorted(store, key=lambda x:x[1])
            #for x in store:
            #    print(x)

            for i,rv in tqdm(zip(np.linspace(0,1,dim),[x[0] for x in store])):
                color=plt.cm.copper(i)
                #print(rv)
                ax.plot(sk.times,1./0.001*project(sk.voft,rv),c=color,alpha=0.8)
            #
            # axs[2].legend(loc="lower left",fontsize=14,ncol=2,handlelength=3.86)
            ax.text(0.05,0.75,'$q='+str(q)+'$\n$\chi_1=\chi_2=0.8$\n right-left',transform=ax.transAxes,linespacing=1.4)
            # for ax in axs:
            ax.set_xlim(-50,50)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            #
            # for ax in axs[:-1]:
            #     ax.set_xticklabels([])
            # for ax in [axs[-1]]:
            ax.set_xlabel("$t\;\;[M]$")
            # for ax,d in zip(axs,["x","y","z","v_k"]):
            #     ax.set_ylim(-1.4,1.4)
            ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat n} \;\;[0.001c]$")

            figs.append(fig)

        return figs

    @classmethod
    @plottingstuff
    def spinaligned(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        figs=[]
        L=0.7
        H=0.3
        S=0.05

        for q in [1,0.5]:

            fig = plt.figure(figsize=(6,6))
            axs = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]

            chimag=0.8
            sks=[ bhbin(q=q,chi1=[0,0,chimag],chi2=[0,0,chimag],t_ref=-100), bhbin(q=q,chi1=[0,0,-chimag],chi2=[0,0,-chimag],t_ref=-100),
            bhbin(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag],t_ref=-100),
            bhbin(q=q, chi1=[0,0,-chimag],chi2=[0,0,chimag],t_ref=-100)]


            labels=["up-up","down-down","up-down","down-up"]
            dashes=['',[15,5],[8,5],[2,2]]
            cols=['C0','C1','C2','C3']
            for sk,l,d,c in zip(sks,labels,dashes,cols):
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=0.4,lw=1,c=c)
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=0.4,lw=1,c=c)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=0.4,lw=1,c=c)
                axs[2].plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=1,lw=2,c=c,dashes=d,label=l)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=0.4,lw=1,c=c)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=1,lw=2,c=c,dashes=d)

                print(l,convert.kms(sk.kick))

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
                ax.set_ylim(-1.4,1.4)
                ax.set_ylabel("$\mathbf{v}(t) \cdot \mathbf{\hat "+d+"} \;\;[0.001c]$")

            figs.append(fig)

        return figs



    @classmethod
    @plottingstuff
    def hangupErad(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        figs=[]
        L=0.7
        H=0.35
        S=0.05

        for q in [0.5]:

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_axes([0,0,L,H])

            chimag=0.8
            sks=[ bhbin(q=q,chi1=[0,0,chimag],chi2=[0,0,chimag]),
                bhbin(q=q,chi1=[0,0,-chimag],chi2=[0,0,-chimag]),
                bhbin(q=q,chi1=[0,0,chimag],chi2=[0,0,-chimag]),
                bhbin(q=q, chi1=[0,0,-chimag],chi2=[0,0,chimag]),
                bhbin(q=q, chi1=[0,0,0],chi2=[0,0,0])]


            labels=["$\chi_i\!=\!0.8$, up-up","$\chi_i\!=\!0.8$, down-down","$\chi_i\!=\!0.8$, up-down","$\chi_i\!=\!0.8$, down-up","$\chi_i=0$"]
            dashes=['',[15,5],[8,5],[2,2],[0.5,1]]
            cols=['C0','C1','C2','C3','black']
            for sk,l,d,c in zip(sks,labels,dashes,cols):
                ax.plot(sk.times,sk.Eoft,alpha=0.4,lw=1,c=c)
                ax.plot(sk.times,sk.Eoft,alpha=1,lw=2,c=c,dashes=d,label=l)
            #
            ax.legend(loc="upper left",fontsize=11,handlelength=5.5)
            ax.text(0.8,0.1,'$q='+str(q)+'$',transform=ax.transAxes,linespacing=1.4)
            # for ax in axs:
            ax.set_xlim(-50,50)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())

            # for ax in axs[:-1]:
            #     ax.set_xticklabels([])
            # for ax in [axs[-1]]:
            ax.set_xlabel("$t\;\;[M]$")
            # for ax,d in zip(axs,["x","y","z","v_k"]):
            ax.set_ylim(0,0.08)
            ax.set_ylabel("${E(t)} \;\;[M]$")

            figs.append(fig)

        return figs


    @classmethod
    @plottingstuff
    def leftright(self):
        ''' Attempt to reproduce Fig 4  in Brugmann+ 2008, their alpha series'''

        figs=[]
        L=0.7
        H=0.3
        S=0.05
        Z=0.35

        for q in [1,0.5]:

            fig = plt.figure(figsize=(6,6))
            axs = [fig.add_axes([0,-i*(S+H),L,H]) for i in [0,1,2,3]]
            axi = fig.add_axes([0,-1*(S+H)-S-Z*H,Z*L*1.2,Z*H])

            chimag1=0.8
            chimag2=0.8
            sks=[ bhbin(q=q,chi1=[chimag1,0,0],chi2=[chimag2,0,0],t_ref=-125),
            bhbin(q=q,chi1=[-chimag1,0,0],chi2=[-chimag2,0,0],t_ref=-125),
            bhbin(q=q,chi1=[chimag1,0,0],chi2=[-chimag2,0,0],t_ref=-125),
            bhbin(q=q, chi1=[-chimag1,0,0],chi2=[chimag2,0,0],t_ref=-125)]


            labels=["right-right","left-left","right-left","left-right"]
            dashes=['',[15,5],[8,5],[2,2]]
            cols=['C0','C1','C2','C3']
            for sk,l,d,c in zip(sks,labels,dashes,cols):
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=0.4,lw=1,c=c)
                axs[0].plot(sk.times,1./0.001*project(sk.voft,[1,0,0]),alpha=1,lw=2,c=c,dashes=d)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=0.4,lw=1,c=c)
                axs[1].plot(sk.times,1./0.001*project(sk.voft,[0,1,0]),alpha=1,lw=2,c=c,dashes=d,label=l)
                for ax in [axs[2],axi]:
                    ax.plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=0.4,lw=1,c=c)
                    ax.plot(sk.times,1./0.001*project(sk.voft,[0,0,1]),alpha=1,lw=2,c=c,dashes=d)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=0.4,lw=1,c=c)
                axs[3].plot(sk.times,1./0.001*project(sk.voft,sk.kickdir),alpha=1,lw=2,c=c,dashes=d)



                print(l,convert.kms(sk.kick),convert.kms(sk.kickcomp))

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

            axi.set_xlim(-450,-150)
            axi.set_ylim(-0.02,0.02)
            axi.yaxis.tick_right()

            axi.set_xticks([-400,-300,-200])
            axi.set_yticks([-0.015,0,0.015])
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
    def testoptimizer(self):

        allfig=[]

        for j in range(16):
            print(j)
            fig = plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7])
            q=np.random.uniform(0.5,1)
            chi1m=np.random.uniform(0,0.8)
            chi2m=np.random.uniform(0,0.8)
            theta1=np.arccos(np.random.uniform(-1,1))
            theta2=np.arccos(np.random.uniform(-1,1))
            deltaphi=np.random.uniform(-np.pi,np.pi)
            chi1,chi2=convert.anglestocoords(chi1m,chi2m,theta1,theta2,deltaphi,1.)
            alpha_vals=np.linspace(-np.pi,np.pi,100)

            tref_vals=[None]+list(np.linspace(-3500,-500,4))
            for i,t_ref in enumerate(tref_vals):
                print(j,t_ref)
                color=plt.cm.copper(i/len(tref_vals))
                ok=optkick(q=q,chi1=chi1,chi2=chi2,t_ref=t_ref)
                ax.plot(alpha_vals,convert.kms(ok.phasesur(alpha_vals)),color=color)

            ax.plot(alpha_vals,convert.kms(ok.phasefit(alpha_vals)),color='dodgerblue',lw=2.5,dashes=[8,3])


            #fitmin,fitmax = ok.find(flag='fit')
            #ax.axhline(fitmin,c='C0',ls='dotted')
            #ax.axhline(fitmax,c='C0',ls='dotted')
            #fitmin,fitmax = ok.find(flag='sur')
            #ax.axhline(fitmin,c='C1',ls='dotted')
            #ax.axhline(fitmax,c='C1',ls='dotted')

            #
            #     kick_vals=[]
            #     for alpha in alpha_vals:
            #         sk = bhbin(q=1 , chi1=[chimag*np.cos(alpha),chimag*np.sin(alpha),0],chi2=[-chimag*np.cos(alpha),-chimag*np.sin(alpha),0])
            #         kick_vals.append(convert.kms(sk.kickcomp[-1]))
            #
            #     ax.plot(alpha_vals,kick_vals,color=color)
            #     #ax.plot(alpha_vals,2725*np.cos(0.98*np.pi+alpha_vals))
            #
            #
            #ax.set_xlim(-np.pi,2.*np.pi)
            # ax.set_ylim(-3000,3000)
            ax.set_xlabel("$\\alpha$")
            ax.set_ylabel("$v_k\;\;[{\\rm km/s}]$")

            # ax.set_ylabel("$\mathbf{v}(t)\cdot \hat\mathbf{z} \;\;[{\\rm km/s}]$")
            ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
            ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
            #
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            #ax.legend(bbox_to_anchor=(1.05, 1),loc=2,bbox_transform=ax.transAxes, borderaxespad=0.)

            ax.text(1.05,1,'$q='+str(round(q,2))+'$\n$\chi_1='+str(round(chi1m,2))+'$\n$\chi_2='+str(round(chi2m,2))+'$\n$\\theta_1='+str(round(theta1,2))+'$\n$\\theta_2='+str(round(theta2,2))+'$\n$\Delta\Phi='+str(round(deltaphi,2))+'$',verticalalignment='top',transform=ax.transAxes)

            allfig.append(fig)
        return allfig


    @classmethod
    @plottingstuff
    def kickdistr(self):

        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.7])



        f=h5py.File('kickdistr.h5')
        dim=int(1e5)
        kicks=[]
        try:
            kicks=list(f[f.keys()[0]])
            dim=max(0, dim - len(kicks))
        except:
            f.create_dataset('kicks', data=kicks)
        f.close()
        print("dime", dim)

        if dim>0:

            for i in tqdm(range(dim)):
                #print(i,dim)
                q=np.random.uniform(0.5,1)

                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]

                sk= bhbin(q=q,chi1=chi1,chi2=chi2)
                kicks.append(sk.kick)

            f=h5py.File('kickdistr.h5')
            del f['kicks']
            f.create_dataset('kicks', data=kicks)
            f.close()

        print(convert.kms(max(kicks)))

        ax.hist(kicks,bins=100)
        return fig

    @classmethod
    @plottingstuff
    def normprofiles(self):

        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.7])
        #ax2=fig.add_axes([1,0,0.7,0.7])


        levels = np.linspace(0,0.0016,100)
        CS3 = plt.contourf([[0,0],[0,0]], levels, cmap=plt.cm.copper,extend='max')
        plt.clf()


        fig = plt.figure(figsize=(6,6))
        ax=fig.add_axes([0,0,0.7,0.7])
        #ax2=fig.add_axes([1,0,0.7,0.7])


        if not os.path.isfile("normprofiles.pkl"):
            #data1=[]
            #data2=[]
            #data3=[]
            #data4=[]
            data=[]
            for i in tqdm(range(200)):
                #print(i,dim)
                q=np.random.uniform(0.5,1)

                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                r = 0.8*(np.random.uniform(0,1))**(1./3.)
                chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]

                sk= bhbin(q=q,chi1=chi1,chi2=chi2)

                phi = np.random.uniform(0,2*np.pi)
                theta = np.arccos(np.random.uniform(-1,1))
                randomdir= [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ]

                #data1.append(sk.times)
                #data2.append(project(sk.voft/sk.kick,sk.kickdir))
                #data2.append(project(sk.voft/sk.kick,sk.kickdir))
                #data3.append(project(sk.voft/np.dot(sk.kickcomp,randomdir),randomdir))
                #data4.append(sk.kick)

                data.append([sk.kick, project(sk.voft/sk.kick,sk.kickdir)])


            #with open("normprofiles.pkl", 'wb') as f: pickle.dump([data1,data2,data3,data4], f)
            with open("normprofiles.pkl", 'wb') as f: pickle.dump(data, f)
        #with open("normprofiles.pkl", 'rb') as f: data1,data2,data3,data4 = pickle.load(f)
        with open("normprofiles.pkl", 'rb') as f: data = pickle.load(f)

        times=bhbin().times
        for d in tqdm(data):

            ax.plot(times,d[1],alpha=0.3, c= plt.cm.copper(d[0]/0.0016),lw=0.9)
        axcb = fig.add_axes([0.75,0,0.05,0.7])
        cb = fig.colorbar(CS3,cax=axcb,boundaries=np.linspace(0,0.016,100),ticks=np.linspace(0,0.0016,9))

        #for d1,d2,d3,d4 in tqdm(zip(data1,data2,data3,data4)):

            #ax.plot(d1,d2,alpha=0.2, c= plt.cm.copper(d4/0.001))
            #ax2.plot(d1,d3,alpha=0.2, c= plt.cm.copper(d4/0.001))

        ax.plot(times,scipy.stats.norm.cdf(times, loc=10, scale=8),dashes=[10,4],c='dodgerblue',lw=1.5)

        ax.set_xlim(-50,50)
        #ax2.set_xlim(-50,50)

        return fig


    @classmethod
    @plottingstuff
    def explore(self):

        figs=[]

        for i in tqdm(range(int(1000))):
            #print(i,dim)
            q=np.random.uniform(0.5,1)

            phi = np.random.uniform(0,2*np.pi)
            theta = np.arccos(np.random.uniform(-1,1))
            r = 0.8*(np.random.uniform(0,1))**(1./3.)
            chi1= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]
            phi = np.random.uniform(0,2*np.pi)
            theta = np.arccos(np.random.uniform(-1,1))
            r = 0.8*(np.random.uniform(0,1))**(1./3.)
            chi2= [ r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta) ]


            sk= bhbin(q=q,chi1=chi1,chi2=chi2)

            fig= plt.figure(figsize=(6,6))
            ax=fig.add_axes([0,0,0.7,0.7])
            ax.plot(sk.times,project(sk.voft/sk.kick,sk.kickdir),c= plt.cm.copper(sk.kick/0.001))

            ax.set_xlim(-50,50)
            ax.set_xlabel("$t/M$")
            ax.set_ylabel("$\mathbf{v(t)}/v_k \cdot \mathbf{\hat{v}_k}$")
            ax.set_title('$q='+str(round(q,2))+'\;\;\chi_1='+str([round(x,2) for x in chi1])+'\;\;\chi_2='+str([round(x,2) for x in chi2])+'\;\;v_k='+str(round(sk.kick,5))+'$')


            figs.append(fig)

        return figs



########################################
if __name__ == "__main__":

    #plots.lineofsight()

    plots.alphaprof()
    #plots.centerofmass()
    #plots.alpharies()
    #for t_ref in np.linspace(-4500,-100,10):
    #    print(bhbin(q=1,chi1=[0.8,0,0],chi2=[-0.8,0,0],t_ref=t_ref).kick)
    #print(convert.kms(bhbin(q=0.5,chi1=[0,0,0],chi2=[0,0,0],t_ref=-100).kick))
    #plots.hangupErad()

    #plots.leftright()
    #plots.hangup()
    #plots.nonspinning()
    #plots.residuals()
    #plots.profiles()
    #plots.centerofmass()
    #plots.alphaseries()
    #plots.centerofmass()
    #plots.nospinprofiles()
    #print(fitkick(0.5,1,-0.5,0,0,0,0))
    #print(fitkick2(0.5,1,-0.5,0,0,0,0))
    #print(convert.kms(bhbin(q=0.5).kick))
    #plots.kickdistr()
    #plots.normprofiles()
    #plots.testoptimizer()
    #plots.explore()
    #chi1=[0.4,-0.5,0.9]
    #chi2=[-0.4,0.5,-0.9]
    #print(chi1,chi2)
    #print(convert.anglestocoords(*convert.coordstoangles(chi1,chi2)))
    #plots.nospinprofiles()
    #chi1mag,chi2mag,theta1,theta2,deltaphi,phi1=0.4,0.6,1,1,0,-1.57
    #print(convert.coordstoangles(*convert.anglestocoords(chi1mag,chi2mag,theta1,theta2,deltaphi,phi1)))

    #print(scipy.optimize.fminbound(lambda bigTheta: -fitkick(0.8,0.7,0.7,1.57,1/,1,bigTheta), -np.pi, np.pi,))

    #print(scipy.optimize.fminbound(lambda bigTheta: -bhbin(q=0.8,chi1=), -np.pi, np.pi,))

    #print(orbitaloptimizer(0.8,[0.1,0.1,0],[-0.7,0,0]))
    #
    # sur=surrogate()
    # print(sur)
    # sur2=surrogate()
    # print(sur2)

surrkick
==========

## Black-hole kicks from numerical-relativity surrogate models

surrkick is a python module to extract radiate energy and momenta from waveform approximants. 
The present version of the code used the numerical-relativity surrogate model NRSur7dq2.

When using `surrkick` in any published work, please cite the paper describing its implementation:

- *Black-hole kicks from numerical-relativity surrogate models.*
Davide Gerosa, Francois Hebert.
[arXiv:1802.XXXXX](https://arxiv.org/abs/arXiv:1802.XXXXX)

More info on the code available in Sec. 5 of our paper and at [davidegerosa.com/surrkick](https://davidegerosa.com/surrkick])
Surrkick is distributed through the [Python Package index](https://pypi.python.org/pypi/surrkick)
and here on [github](github.com/dgerosa/surrkick).

### Installation

Surrkick is a python module, uploaded to the Python Package Index. Installation is as easy as

    pip install surrkick
  
The SXS surrogate model NRSur7dq2 and a few other dependencies will be installed together with surrkick.
You can try some functionalities with

    import surrkick
    surrkick.plots.minimal()

### Main functions

The core of the code consists in a single class, `surrkick`, whose methods allow to extract radiated energy, linear momentum (kikcs) 
and angular momentum from the underlying wavefeform approximant

- `sur()`: Instance of the surrogate class from NRSur7dq2
- `q`: Binary mass ratio 0.5<=q<=1 (Default: q=1).
- `chi1`: Spin vector of the heavier BH at t_ref (Default chi1=[0,0,0]).
- `chi2`: Spin vector of the lighter BH at t_ref (Default chi2=[0,0,0]).
- `t_ref`: Reference time -4500<=t_ref/M<=-100 (Default t_ref/M=100).
- `times`: Time nodes -4500<=t_i/M<=-100                                                                            
- `lmax`: Largest available l-mode (lmax=4 in NRSur7dq2).                                                            
- `h(l,m)`: Modes of the complex GW strain.              
- `hdot(l,m)`: Modes of the time derivative of the GW strain.                           
- `dEdt`: Energy flux dE/dt.    
- `Eoft`: Radiated energy profile E(t).                                                                              
- `Erad`: Total radiated energy.                                                               
- `dPdt`: Linear momentum flux dP/dt.
- `Poft`: Radiated linear momentum profile P(t).                                                       
- `Prad`: Total radiated linear momentum.                               
- `voft`: Recoil velocity profile v(t).               
- `kickcomp`: Kick velocity, vector.              
- `kick`: Kick velocity, magnitude vk                                                                             
- `kickdir`: Kick velocity, unit vector.                                     
- `dJdt`: Angular momentum flux dJ/dt. 
- `Joft`: Radiated angular momentum profile J(t).                                                   
- `Jrad`: Total radiated angular momentum.                                   
- `xoft`: Center-of-mass trajectory x(t).   

The class plots contains script to reproduce all figures and results in 
[arXiv:1802.XXXXX](https://arxiv.org/abs/arXiv:1802.XXXXX). 
You can explore its methods with, e.g. `help(surrkick.plots)`. In particular, `surrkick.plots.minimal()` is:
    
    import surrkick
    import matplotlib.pyplot as plt
    sk=surrkick.surrkick(q=0.5,chi1=[0.8,0,0],
        chi2=[-0.8,0,0])
    print "vk/c=", sk.kick
    plt.plot(sk.times,sk.voft[:,0],label="x")
    plt.plot(sk.times,sk.voft[:,1],label="y")
    plt.plot(sk.times,sk.voft[:,2],label="z")
    plt.plot(sk.times,surrkick.project(sk.voft,
        sk.kickdir),label="vk")
    plt.xlim(-100,100)
    plt.legend()
    plt.show()


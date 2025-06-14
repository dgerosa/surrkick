# surrkick

Code and data release supporting

- *Black-hole kicks from numerical-relativity surrogate models.*
Davide Gerosa, Francois Hebert, Leo C. Stein.
Phys. Rev. D 97, 104049 (2018).
[arXiv:1802.04276](https://arxiv.org/abs/arXiv:1802.04276)


## Credits

You are welcome to use our software/data in your research. We kindly ask you to cite the paper above. If you want to cite this data release specifically, the DOI code is: [![DOI](https://zenodo.org/badge/119469724.svg)](https://zenodo.org/badge/latestdoi/119469724)

## Installation

`surrkick` is a python module, uploaded to the [Python Package index](https://pypi.python.org/pypi/surrkick). Installation is as easy as

    pip install surrkick
  
The SXS surrogate model NRSur7dq2 and a few other dependencies will be installed together with surrkick. If you don't have it already, you might need to manually install numpy beforehand (that's `pip install numpy`).
You can try some functionalities with

    import surrkick
    surrkick.plots.minimal()

## Main functions

The core of the code consists of a single class, surrkick, whose methods allow to extract radiated energy, linear momentum (kicks) and angular momentum from the underlying waveform approximant. The main methods are:

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
- `Moft`: Mass profile M(t).
- `Mfin`: Mass of the remnant BH.
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

The class `plots` contains scripts to reproduce all figures and results in  [arXiv:1802.04276](https://arxiv.org/abs/arXiv:1802.04276). You can explore its methods with, e.g. `help(surrkick.plots)`. 

## Test

The source code for the `surrkick.plots.minimal()` method mentioned above is
    
    import surrkick
    import matplotlib.pyplot as plt
    sk=surrkick.surrkick(q=0.5,chi1=[0.8,0,0],
        chi2=[-0.8,0,0])
    print 'vk/c=', sk.kick
    plt.plot(sk.times,sk.voft[:,0],label='x')
    plt.plot(sk.times,sk.voft[:,1],label='y')
    plt.plot(sk.times,sk.voft[:,2],label='z')
    plt.plot(sk.times,surrkick.project(sk.voft,
        sk.kickdir),label='vk')
    plt.xlim(-100,100)
    plt.legend()
    plt.show()

If you try it, you should get a printout that reads `vk/c= 0.00384...` and this plot:
![minimal](https://user-images.githubusercontent.com/7237041/35894834-7f84c500-0b69-11e8-99bd-bc4faa738fda.png)

## Movies

Here are some animations produced with `surrkick`. Also available in this [Youtube playlist](https://www.youtube.com/watch?v=a_StJoMvCSU&list=PLVjP4QK1oHulLGS1qKonkmWeQvMEADGYs&index=2&t=0s)


https://github.com/user-attachments/assets/06724bb2-c7ea-43e8-aec0-4e033d1e2ac6

https://github.com/user-attachments/assets/d35acea3-f5f9-4996-bb74-ef760640f5b9

https://github.com/user-attachments/assets/24ee917d-e987-4baf-a231-853c59737757



import sys, os
src_path = "../HNL/"
sys.path.append(src_path)
src_path = "../../"
sys.path.append(src_path)
import numpy as np
from src.foresee import Foresee, Utility, Model
from matplotlib import pyplot as plt
import pickle
from scipy.special import factorial,gammaln
from HNLCalc import * 
from tqdm import tqdm
from scipy import stats
import matplotlib.colors as colors


def init_llp(llp,energy = "14"):

    if llp == 'DarkPhoton': return init_DarkPhoton(energy = energy)

    elif llp == 'U(1)B-L': return init_U1BL(energy=energy)

def init_HNL(mixing,majorana,energy = "14") : 

    """
        Initialize the FORESEE instance for the HNL Model.
    
        Parameters
        ----------
        energy: str
            C.O.M. energy
        majorana: bool
            determines if HNLs are Majorana or Dirac
        mixing: tuple
            defines the HNL coupling ratio
    
        Returns
        -------
            FORESEE object
    """

    modelname="HNL"
    
    model = Model(modelname, path="../HNL/HNL-parameter/")

    hnl = HeavyNeutralLepton(ve=mixing[0], vmu=mixing[1], vtau=mixing[2], majorana = majorana,modelpath=model.modelpath)
    
    hnl.set_generators(
    generators_light = ['EPOSLHC'], # ['EPOSLHC', 'SIBYLL', 'QGSJET']
    generators_heavy = ['NLO-P8'] # ['NLO-P8','NLO-P8-Max', 'NLO-P8-Min'] 
)
        
    production_channels = []
    
    for label, pid0, pid1, br, generator, description in hnl.get_channels_2body():
        if True:
            #print ('include:', description)
            model.add_production_2bodydecay(
                label = label,
                pid0 = pid0,
                pid1 = pid1,
                br = br,
                generator = generator,
                energy = energy,
                nsample = 25,
            )
            production_channels.append([label, None, description])
    
    for label, pid0, pid1, pid2,br,generator,integration,description in hnl.get_channels_3body():
        
        if True:
            #print ('include:', description)
            model.add_production_3bodydecay(
                label = label,
                pid0 = pid0,
                pid1 = pid1,
                pid2 = pid2,
                br = br,
                generator = generator,
                energy = energy,
                nsample = 25,
                integration = integration,
            )
            production_channels.append([label, None, description])
    
    hnl.get_br_and_ctau()
    
    model.set_ctau_1d(
        filename=f"model/ctau/ctau.txt",
        coupling_ref=1 
    )
    
    modes,finalstates,filenames = hnl.set_brs()
    
    model.set_br_1d(
        modes=modes,
        finalstates=finalstates,
        filenames=filenames
    )
    
    foresee = Foresee()
    foresee.set_model(model=model)

    return foresee

def init_U1BL(energy = "14"):

    """
        Initialize the FORESEE instance for the U(1) B-L Model.
    
        Parameters
        ----------
        energy: str
            C.O.M. energy
    
        Returns
        -------
            FORESEE object
    """


    modelname="U(1)B-L"
    
    model = Model(modelname, path="../U(1)B-L/")

    model.add_production_2bodydecay(
        pid0 = "111",
        pid1 = "22",
        br = "2*0.99 * (coupling/0.303)**2 * pow(1.-pow(mass/self.masses('pid0'),2),3)",
        generator = ['EPOSLHC'],
        energy = energy,
        nsample = 100
    )

    model.add_production_2bodydecay(
        pid0 = "221",
        pid1 = "22",
        br = "2*0.25*0.39  * (coupling/0.303)**2 * pow(1.-pow(mass/self.masses('pid0'),2),3)",
        generator = ['EPOSLHC'],
        energy = energy,
        nsample = 100, 
    )

    masses_brem = [ 
    0.01  ,  0.0126,  0.0158,  0.02  ,  0.0251,  0.0316,  0.0398,
    0.0501,  0.0631,  0.0794,  0.1   ,  0.1122,  0.1259,  0.1413,
    0.1585,  0.1778,  0.1995,  0.2239,  0.2512,  0.2818,  0.3162,
    0.3548,  0.3981,  0.4467,  0.5012,  0.5623,  0.6026,  0.631 ,
    0.6457,  0.6607,  0.6761,  0.6918,  0.7079,  0.7244,  0.7413,
    0.7586,  0.7762,  0.7943,  0.8128,  0.8318,  0.8511,  0.871 ,
    0.8913,  0.912 ,  0.9333,  0.955 ,  0.9772,  1.    ,  1.122 ,
    1.2589,  1.4125,  1.5849,  1.7783,  1.9953,  2.2387,  2.5119,
    2.8184,  3.1623,  3.9811,  5.0119,  6.3096,  7.9433, 10.    
]

    model.add_production_direct(
        label = "Brem",
        energy = energy,
        condition = ["p.pt<1"],
        coupling_ref=0.303,
        masses = masses_brem,
    )

    model.set_ctau_1d(
    filename="model/ctau.txt", 
)

    decay_modes = ["e_e", "mu_mu", "nu_nu", "Hadrons"]     
    model.set_br_1d(
        modes = decay_modes,
        finalstates=[[11,-11], [13,-13], [12,-12], None],
        filenames=["model/br/"+mode+".txt" for mode in decay_modes],
    )

    foresee = Foresee(path=src_path)
    foresee.set_model(model=model)

    return foresee
    
def init_DarkPhoton(energy = "14") : 

    """
        Initialize the FORESEE instance for the Dark Photon Model.
    
        Parameters
        ----------
        energy: str
            C.O.M. energy
    
        Returns
        -------
            FORESEE object
    """

    
    modelname="DarkPhoton"
    model = Model(modelname, path="../DarkPhoton/")
    
    model.add_production_2bodydecay(
        pid0 = "111",
        pid1 = "22",
        br = "2.*0.99 * coupling**2 * pow(1.-pow(mass/self.masses('pid0'),2),3)",
        generator = ['EPOSLHC'],
        energy = energy,
        nsample = 100,
    )
    
    model.add_production_2bodydecay(
        pid0 = "221",
        pid1 = "22",
        br = "2.*0.39 * coupling**2 * pow(1.-pow(mass/self.masses('pid0'),2),3)",
        generator = ['EPOSLHC'],
        energy = energy,
        nsample = 100, 
    )
    
    model.add_production_mixing(
        pid = "113",
        mixing = "coupling * 0.3/5. * self.masses('pid')**2/abs(mass**2-self.masses('pid')**2+self.masses('pid')*self.widths('pid')*1j)",
        generator = ['EPOSLHC'],
        energy = energy,
    )
    
    masses_brem = [ 
        0.01  ,  0.0126,  0.0158,  0.02  ,  0.0251,  0.0316,  0.0398,
        0.0501,  0.0631,  0.0794,  0.1   ,  0.1122,  0.1259,  0.1413,
        0.1585,  0.1778,  0.1995,  0.2239,  0.2512,  0.2818,  0.3162,
        0.3548,  0.3981,  0.4467,  0.5012,  0.5623,  0.6026,  0.631 ,
        0.6457,  0.6607,  0.6761,  0.6918,  0.7079,  0.7244,  0.7413,
        0.7586,  0.7762,  0.7943,  0.8128,  0.8318,  0.8511,  0.871 ,
        0.8913,  0.912 ,  0.9333,  0.955 ,  0.9772,  1.    ,  1.122 ,
        1.2589,  1.4125,  1.5849,  1.7783,  1.9953,  2.2387,  2.5119,
        2.8184,  3.1623,  3.9811,  5.0119,  6.3096,  7.9433, 10.    
    ]
    
    model.add_production_direct(
        label = "Brem",
        energy = energy,
        condition = ["p.pt<1"],
        coupling_ref=1,
        masses = masses_brem,
    )
    
    masses_dy = [1.5849, 1.7783, 1.9953,2.2387, 2.5119, 2.8184, 3.1623, 3.9811, 5.0119, 6.3096, 7.9433, 10.]
    model.add_production_direct(
       label = "DY",
       energy = energy,
       coupling_ref=1,
       masses = masses_dy,
       condition='True',
    )
    
    model.set_ctau_1d(
        filename="model/ctau.txt", 
    )
    
    decay_modes = ["e_e", "mu_mu", "pi+_pi-", "pi0_gamma", "pi+_pi-_pi0", "K_K"] 
    model.set_br_1d(
        modes = decay_modes,
        finalstates=[[11,-11], [13,-13], [221,-211], [111,22], None, [321,-321]],
        filenames=["model/br/"+mode+".txt" for mode in decay_modes],
    )
    
    foresee = Foresee(path=src_path)
    foresee.set_model(model=model)

    return foresee

def tot_energy(unweighted): 

    """
        Gets sample of total energy measurements, including calorimeter uncertainty, from event sample.
    
        Parameters
        ----------
        unweighted: list
            List of event samples generated by foresee.write_events 
    
        Returns
        -------
            list of total energy measurements for each sample
    """


    tot_energy = []

    weights = []

    for event in unweighted: 
    
        pp, pm = event[4]
    
        Ep, Em = pp.e, pm.e

        Et = Ep + Em
    
        dEt = 0.1*np.sqrt(Et) #assuming calorimeter uncertainty ~10% / sqrt(E) 
        
        while True:
            
            Et_samp= np.random.normal(Et,dEt)

            if Et_samp >=0: break
        
        tot_energy.append(Et_samp)

    return tot_energy

def tot_energy_mupi(unweighted): 

    tot_energy = []

    weights = []

    for event in unweighted: 
    
        pp, pm = event[4]
    
        Ep, Em = pp.e, pm.e

        dEp, dEm = 0.2*Ep , 0.2*np.sqrt(Em)
        
        while True:
            
            Ep_samp, Em_samp = np.random.normal(Ep,dEp),np.random.normal(Em,dEm)

            if Ep_samp >=0 and Em_samp >=0: break
    
        tot_energy.append(Ep_samp+Em_samp)

    return tot_energy

def inv_mass(unweighted):

    """
        Gets sample of invariant mass measurements, including experimental uncertainty, from event sample.
    
        Parameters
        ----------
        unweighted: list
            list of event samples generated by foresee.write_events
    
        Returns
        -------
            list of invariant mass measurements for each sample
    """

    
    inv_mass = []

    weights = []

    for event in unweighted: 
    
        pp, pm = event[4]
    
        Ep, Em = pp.e, pm.e

        dEp, dEm = 0.2*Ep , 0.2*Em #assuming 20% uncertainty on individual energies
    
        pp_3D, pm_3D = np.array([pp[0],pp[1],pp[2]]), np.array([pm[0],pm[1],pm[2]]), 
    
        theta = np.arccos(np.dot(pp_3D,pm_3D)/(pp.p*pm.p))
    
        dth = 0.1*theta  #assuming a 10% openening angle uncertainty
        
        while True:
            
            Ep_samp, Em_samp = np.random.normal(Ep,dEp),np.random.normal(Em,dEm)

            if Ep_samp >=0 and Em_samp >=0: break
        
        while True:
            theta_samp = np.random.normal(theta,dth)
            if theta_samp >=0: break

        m = np.sqrt(2*Ep_samp*Em_samp*(1-np.cos(theta_samp)))
    
        inv_mass.append(m)

    return inv_mass

def inv_mass_mupi(unweighted): 

    inv_mass = []

    weights = []

    for event in unweighted: 
    
        pp, pm = event[4]
    
        Ep, Em = pp.e, pm.e
    
        dEp, dEm = 0.2*Ep , 0.2*np.sqrt(Em)
    
        pp_3D, pm_3D = np.array([pp[0],pp[1],pp[2]]), np.array([pm[0],pm[1],pm[2]]), 
    
        theta = np.arccos(np.dot(pp_3D,pm_3D)/(pp.p*pm.p))
    
        dth = 250e-6
        
        while True:
            
            Ep_samp, Em_samp = np.random.normal(Ep,dEp),np.random.normal(Em,dEm)

            if Ep_samp >=0 and Em_samp >=0: break
        
        while True:
            theta_samp = np.random.normal(theta,dth)
            if theta_samp >=0: break

        m = np.sqrt(2*Ep_samp*Em_samp*(1-np.cos(theta_samp)))
    
        inv_mass.append(m)

    return inv_mass
    
def pseudorapidity(unweighted): 
    
    """
        Gets sample of pseudorapidity measurements, including experimental uncertainty, from event sample.
    
        Parameters
        ----------
        unweighted: list
            list of event samples generated by foresee.write_events
    
        Returns
        -------
            list of pseudorapidity measurements measurements for each sample
    """


    etas = []

    weights = []

    for event in unweighted: 
    
        position = event[1]

        x, y, z = position.x,position.y,position.z + 650

        dx, dy = 800e-6, 20e-6 #assuming 20um uncertianty in magnet bending direction and 800um in the other and perfect precision in z

        x_samp, y_samp = np.random.normal(x,dx), np.random.normal(y,dy)
        
        theta = np.arcsin(np.sqrt(x_samp**2 + y_samp**2)/z)
        
        eta = -np.log(np.tan(theta/2))
    
        etas.append(eta)

    return etas


def get_hist(sample, measurement,bins,weighted=True,plot=False):
    
    """
        Gets histogram of measurements, from a given event sample.
    
        Parameters
        ----------
        sample: list
            list of event samples generated by foresee.write_events
        measurement: str in ['E','eta','eta+E']
            determines which measurement to bin
        bins: array of bin edges or int
            specifies the bin edges or number of bins
        weighted: bool
            if False, counts correspond to number of monte carlo samples in each bin, 
            if True, counts correspond to number of expected events in each bin
        plot: bool
            determines if histogram is plotted
        
        Returns
        -------
            list of bin counts, list of bin_edges, (fig,ax) [if plot == True]
    """


    data = {}


    if plot:
        
        fig,ax = plt.subplots()
        
        ax.set(ylabel = 'events per bin')
    
    if measurement == 'E': 

       
        
        samplex = tot_energy(sample) 
        
        if weighted: 
            counts, bin_edges = np.histogram(samplex,bins=bins,weights=[event[0][0] for event in sample])
            ylabel = 'events per bin'
        else: 
            counts, bin_edges = np.histogram(samplex,bins=bins)
            ylabel = 'samples per bin'
        
        if plot:
            xlabel = r'$E_{e^+ e^-}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.stairs(counts,bin_edges)

    if measurement == 'E-mupi': 

       
        
        samplex = tot_energy_mupi(sample) 
        
        if weighted: 
            counts, bin_edges = np.histogram(samplex,bins=bins,weights=[event[0][0] for event in sample])
            ylabel = 'events per bin'
        else: 
            counts, bin_edges = np.histogram(samplex,bins=bins)
            ylabel = 'samples per bin'
        
        if plot:
            xlabel = r'$E_{\mu\pi}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.stairs(counts,bin_edges)

    if measurement == 'm': 

        samplex = inv_mass(sample) 
        if weighted: 
            counts, bin_edges = np.histogram(samplex,bins=bins,weights=[event[0][0] for event in sample])
            ylabel = 'events per bin'
        else: 
            counts, bin_edges = np.histogram(samplex,bins=bins)
            ylabel = 'samples per bin'
        
        if plot:
            xlabel = r'$m_{e^+ e^-}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.stairs(counts,bin_edges)

    if measurement == 'm-mupi': 

        samplex = inv_mass_mupi(sample) 
        if weighted: 
            counts, bin_edges = np.histogram(samplex,bins=bins,weights=[event[0][0] for event in sample])
            ylabel = 'events per bin'
        else: 
            counts, bin_edges = np.histogram(samplex,bins=bins)
            ylabel = 'samples per bin'
        
        if plot:
            xlabel = r'$m_{\mu\pi}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.stairs(counts,bin_edges)

    elif measurement == 'eta': 
        
        samplex = pseudorapidity(sample) 
        if weighted: 
            counts, bin_edges = np.histogram(samplex,bins=bins,weights=[event[0][0] for event in sample])
            ylabel = 'events per bin'
        else: 
            counts, bin_edges = np.histogram(samplex,bins=bins)
            ylabel = 'samples per bin'
        
        if plot:
            xlabel = r'$\eta$'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.stairs(counts,bin_edges)

    elif measurement == 'eta+E': 
        
        samplex, sampley = pseudorapidity(sample), tot_energy(sample)
        if weighted: 
            if plot: counts, xbins, ybins, h = ax.hist2d(samplex,sampley,bins = bins ,weights=[event[0][0] for event in sample], cmap="rainbow")
            else:counts, xbins, ybins = np.histogram2d(samplex,sampley,bins = bins ,weights=[event[0][0] for event in sample])
            
            zlabel = 'events per bin'
            bin_edges = [xbins,ybins]
        else: 
            if plot: counts, xbins, ybins, h= ax.hist2d(samplex,sampley,bins = bins , cmap="rainbow")
            else: counts, xbins, ybins = np.histogram2d(samplex,sampley,bins = bins )
            zlabel = 'samples per bin'
            bin_edges = [xbins,ybins]
            
        if plot:
            xlabel = r'$\eta$'
            ylabel= r'$E_{e^+e^-}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            fig.colorbar(h, ax=ax,label=zlabel,format="%.2f")

    elif measurement == 'm+E-mupi': 
        
        samplex, sampley = inv_mass_mupi(sample), tot_energy_mupi(sample)
        if weighted: 
            if plot: counts, xbins, ybins, h = ax.hist2d(samplex,sampley,bins = bins ,weights=[event[0][0] for event in sample], cmap="rainbow")
            else:counts, xbins, ybins = np.histogram2d(samplex,sampley,bins = bins ,weights=[event[0][0] for event in sample])
            
            zlabel = 'events per bin'
            bin_edges = [xbins,ybins]
        else: 
            if plot: counts, xbins, ybins, h= ax.hist2d(samplex,sampley,bins = bins , cmap="rainbow")
            else: counts, xbins, ybins = np.histogram2d(samplex,sampley,bins = bins )
            zlabel = 'samples per bin'
            bin_edges = [xbins,ybins]
            
        if plot:
            xlabel = r'$m_{\mu\pi}$ (GeV)'
            ylabel= r'$E_{\mu\pi}$ (GeV)'
            ax.set(xlabel=xlabel,ylabel=ylabel)
            fig.colorbar(h, ax=ax,label=zlabel,format="%.2f")
        
    
        
    if plot: return counts, bin_edges, (fig,ax)
    else: return counts, bin_edges


def chi2(test_data,truth_data):

    """
        Calculates chi2 value between two sets of binned data.
    
        Parameters
        ----------
        test_data: list 
            list of counts in each measurement bin for the 'test' model
        truth_data: list 
            list of counts in each measurement bin for the 'truth' model
    
        Returns
        -------
            float
    """
    
    chi2 = 0.0 

    if truth_data.ndim == 1:
        
        for i in range(len(test_data)):
    
            n = truth_data[i]
    
            mu = test_data[i]
            
            if mu == 0 and n != 0 : chi2 += np.inf
                
            elif n==0: chi2 += 2*mu 
                
            else: chi2 += 2 * (mu-n+n*np.log(n/mu))

    if truth_data.ndim == 2:

        nx, ny  = test_data.shape

        for ix in range(nx): 
            for iy in range(ny):

                n = truth_data[ix,iy]
                
                mu = test_data[ix,iy]

                if mu == 0 and n != 0 : chi2 += np.inf
                
                elif n==0: chi2 += 2*mu 
                    
                else: chi2 += 2 * (mu-n+n*np.log(n/mu))
    return chi2

def get_chi2(masses,couplings,scan,truth_data,measurements,nexp,printing=False,plot = True,xlims=None,ylims=None):

    """
        Calculates the minimum chi2 value of a given paramter scan and draws up to 3 sigma contours.
    
        Parameters
        ----------
        masses: array 
            array of mass points in scan
        couplings: array 
            array of coupling points in scan
        scan: array with shape (len(masses),len(couplings))
            an array containing measurement dictionary for every point in parameter scan (with key values ['E','eta', or 'eta-E']
        truth_data: dictionary
            dictionary containing binned counts for each type of measurment ['E','eta', or 'eta-E']
        measurements: list
            list of measurements to include in the chi2 analysis
        nexp: integer
            number of pseudoexperiments to generate
        printing: bool
            decides whether to print delta chi2 for each point
        plot: bool
            decides whether to plot contours
        xlims: tuple
            figure x-limits
        ylims: tuple
            figure y-limits
            
        Returns
        -------
            array of average chi2 at each point in scan, (fig,ax) 
    """

    scan_means = np.empty(scan.shape,dtype='object')

    scan_dist = np.empty(scan.shape,dtype='object')

    nm, neps = scan.shape
    
    for im in tqdm(range(nm)):
        
        for ieps in range(neps): 

            chi_samp = []
    
            for i in range(nexp):
    
                chi = 0.0
                
                for measurement in measurements: 
                        
                    chi +=chi2(scan[im,ieps][measurement], np.random.poisson(truth_data[measurement]))

                chi_samp.append(chi)
                
            if np.inf not in chi_samp:
                
                chi_dist = stats.rv_histogram(np.histogram(chi_samp,bins=40))
                
                scan_dist[im,ieps] = chi_dist
                
                scan_means[im,ieps] = chi_dist.mean()

                

            else: 
                scan_dist[im,ieps] = None

                scan_means[im,ieps] = np.inf

            
    chi_min = np.min(scan_means)

    dchi = scan_means - chi_min

    i_bf = np.where(scan_means == chi_min)
    
    im_bf, ieps_bf = int(i_bf[0]),int(i_bf[1])

    levels = [scan_dist[int(im_bf),int(ieps_bf)].ppf(.68),scan_dist[im_bf,ieps_bf].ppf(.95),scan_dist[im_bf,ieps_bf].ppf(.997)]

    if printing: 

        print(f"\nBest fit chi2: m = {masses[im_bf]:.2f} eps = {couplings[ieps_bf]:.2e}:\t\t{chi_min:.0f}\n")
        for i,delchi in enumerate(levels): print(f" {i+1} sigma: {delchi:.1f}")

        if True:
            print("\n")
            for im in tqdm(range(nm)):
        
                for ieps in range(neps):

                    print(f"m = {masses[im]:.2f} eps = {couplings[ieps]:.2e}:\t\t{dchi[im,ieps]:.0f}")
    
    if plot: 
        
        fig, ax = plt.subplots()
    
        ax.set(xlim=xlims,ylim=ylims,xlabel=r'$m$ (GeV)',ylabel = r'$\epsilon$',xscale = 'linear', yscale = 'linear')
           
        level_labels = [r'$1\sigma$', r'$2\sigma$',r'$3\sigma$']
        
        c = ax.contour(masses,couplings,dchi.T,levels = levels,colors='k',linestyles=['dotted','--','-'])
        
        fmt = {}
        
        for l, s in zip(c.levels, level_labels): fmt[l] = s
        
        ax.scatter(masses[im_bf],couplings[ieps_bf],color='k',marker='x',label=rf'Best Fit: $\chi^2 = {chi_min:.1f}$')
        
        ax.legend()
 
    return scan_means, (fig,ax)
                        

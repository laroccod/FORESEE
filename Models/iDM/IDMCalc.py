import os, random, math, gzip
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
#from os.path import exists
from src.foresee import Utility, Decay, energy_stem
from src.utils.utility import BREM_MASSES
from src.utils.vectors import LorentzArray
from src.utils.utility import BREM_MASSES

# chi_2 spectrum grid: union of the brem/DY A' source grids on the 0.05
# (log theta, log p) lattice (matches files/direct/iDM/<E>.txt.gz).
IDM_PRANGE = [[-7.5, 0.2, 154], [-2, 5, 140]]


class InelasticDarkMatter(Utility, Decay):

    ###############################
    #  Initiate
    ###############################

    def __init__(self, alphaD=1, delta=0, r=0):
        
        self.alphaD = alphaD
        self.delta = delta
        self.r = r
        self.rng = random.Random()
        
        self.masses_brem_aprime = BREM_MASSES
        
        # Dark-photon (A') mass grid for Drell-Yan production. The precomputed
        # chi_2 spectra (files/direct/iDM/<E>.txt.gz, "DY" columns) were generated
        # on this grid; get_dy_masses() maps it to chi_2 masses for the chosen
        # benchmark. DY spectra are only shipped for the high-energy beams.
        self.masses_dy_aprime = [1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184, 3.1623, 3.9811,
            5.0119, 6.3096, 7.9433, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0, 
            50.0, 70.0, 100.0]

        self.masses_mixing_aprime = np.logspace(-2,1,31)
        
    def get_brem_masses(self):
        # chi_2 masses for the bremsstrahlung spectra: m_chi2 = m_{A'} * (1+delta)/r.
        # The A' grid is the shared extended bremsstrahlung grid (src/utils/utility.py),
        # so the chi_2 grid matches the "<m>(Brem_*)" columns of the iDM direct file.
        return [round(m0 / self.r * (1+self.delta), 5) for m0 in BREM_MASSES]

    def get_dy_masses(self):
        # chi_2 masses for the Drell-Yan spectra, same m_{A'} -> m_chi2 mapping.
        return [round(m0 / self.r * (1+self.delta), 5) for m0 in self.masses_dy_aprime]

    def get_mixing_masses(self,):
        return [round(m0 / self.r * (1+self.delta),5)  for m0 in self.masses_mixing_aprime]

    def get_masses(self,):
        masses_combined = self.masses_brem_aprime + self.masses_dy_aprime
        masses_unique=[]
        [masses_unique.append(x) for x in masses_combined if x not in masses_unique]
        return masses_unique
        
#    Approximate expression we used previously (Eq.13 in 1810.01879). Now we use the more complete expression below.
#    def get_ctau(self, m2 ,epsilon=1):
#        alphaEM = 1/137.
#        hbarc = 0.2e-15 # GeV * m
#        map = m2 * self.r / (1+self.delta)
#        term1 = (4 * epsilon**2 * alphaEM * map)/(15 * np.pi)
#        term2 = self.alphaD * self.delta**5 / self.r**5
#        return hbarc/(term1*term2)

#   More complete expression from Eq.B7 in 1911.11346
    def get_ctau(self, m2 ,epsilon=1):
        # constants
        alphaEM = 1/137.
        hbarc = 0.2e-15 # GeV * m
        me = 0.5e-3

        m1 = m2 / (1 + self.delta)
        map = m2 * self.r / (1+self.delta)
        
        # chi2 > chi1 e+ e- can only happen if m2 - m1 > 2*me
        if  m2*self.delta/(1+self.delta) < 2*me:
            return (np.nan)
        
        # define Kallen lambda function
        lmbda = lambda a,b,c: (a-b-c)**2 - 4*b*c
 
        # obtain BR of A' ->e+e-
        # Library layout: direct_darkphoton/ lives directly under Models/iDM/,
        # so this path is relative to the model dir (was "../direct_darkphoton"
        # in foresee-working, where the notebook sat one level deeper in iDM_BP1/).
        filename='model/br_darkphoton/e_e.txt'
        aprime_ee = np.loadtxt(filename).T
        br_interp = interp1d(aprime_ee[0], aprime_ee[1])
        Be = br_interp(map)
        
        # define width of A' (Eq. B6 in 1911.11346)
        gamma = lambda : ((epsilon**2 * alphaEM * map)/(3) * np.sqrt(1 - (4*me**2/map**2)) * (1 + (2*me**2/map**2)) ) / (Be)

        # integration limits
        s1p = lambda s2: m1**2 + me**2 + (1/(2*s2))*( (m2**2 - me**2 - s2)*(m1**2 - me**2 + s2) + (np.sqrt(lmbda(s2,m2**2,me**2)) * np.sqrt(lmbda(s2,m1**2,me**2))) )
        s1m = lambda s2: m1**2 + me**2 + (1/(2*s2))*( (m2**2 - me**2 - s2)*(m1**2 - me**2 + s2) - (np.sqrt(lmbda(s2,m2**2,me**2)) * np.sqrt(lmbda(s2,m1**2,me**2))) )
        s2p = (m2 - me)**2
        s2m = (m1 + me)**2
        
        # define integrand (Eq. B7 in 1911.11346)
        prefactor = (epsilon**2 * self.alphaD * alphaEM) / (16 * np.pi * m2**3)
        asquared = lambda s1,s2: (s1 + s2 - 2*m1*m2 - 2*me**2) * ((m1+m2)**2 + 4*me**2) + 2*(me**2 + m1*m2)**2 - s1**2 - s2**2
    
        dnr = lambda s1,s2: ((m1**2 + m2**2 + 2*me**2 - s1 - s2 - map**2)**2 + map**2*gamma()**2) * 1 # This 1 = Br(A->ee) at mA = 2me
        
        integrand = lambda s1,s2: 4*asquared(s1,s2)/dnr(s1,s2)
        width = integrate.dblquad(integrand, s2m, s2p, s1m, s1p)[0]*prefactor
        ctau = hbarc/width
        return ctau

    def obtain_ctau_br(self, masses=np.logspace(-2,2, 601), epsilon=1):
    
        # set path, make sure it exists
        path = "model/"
        os.makedirs(path,exist_ok = True)
            
        #calculate ctaus
        data = np.array([[mass, self.get_ctau(mass, epsilon)] for mass in masses])
        data = data[~np.isnan(data[:, 1])]
        # save
        filepath = path + "ctau.txt"
        np.savetxt(filepath, data, delimiter=' ')

    def get_X2_spectrum(self, momenta, weights, m0, m1, m2, nsample=10, prange=IDM_PRANGE):
        # Decay A' -> chi1 chi2 (keeping chi2 = p2) and rebin into the
        # (logth, logp) grid. Each loaded A' momentum is decayed nsample times
        # with a random orientation; weights carries one column per loaded A'
        # configuration, and one rebinned weight list is returned per column.
        weights = np.asarray(weights)
        n = len(weights)

        # repeat each A' momentum nsample times, one random orientation each
        p0 = LorentzArray({
            "px": np.repeat(np.asarray(momenta.px), nsample),
            "py": np.repeat(np.asarray(momenta.py), nsample),
            "pz": np.repeat(np.asarray(momenta.pz), nsample),
            "energy": np.repeat(np.asarray(momenta.energy), nsample),
        })
        phi = np.array([self.rng.uniform(-math.pi, math.pi) for _ in range(n * nsample)])
        cos = np.array([self.rng.uniform(-1., 1.) for _ in range(n * nsample)])
        p1, p2 = self.twobody_decay(p0, m0, m1, m2, phi, cos)

        tx = np.arctan(np.asarray(p2.pt) / np.asarray(p2.pz))
        px = np.asarray(p2.p)
        weights_chi2 = np.repeat(weights, nsample, axis=0) / nsample
        cols = []
        for j in range(weights_chi2.shape[1]):
            list_th, list_p, list_w = self.get_hist_list(tx, px, weights_chi2[:, j], prange)
            cols.append(list_w)
        return list_th, list_p, cols

    def obtain_direct_production(self, energy='14', nsample_load=1, nsample=10, prange=IDM_PRANGE):
        # Build the chi_2 direct-production spectra by decaying A' -> chi1 chi2
        # for the chosen benchmark. Reads the consolidated dark-photon (A')
        # tables (files/direct/DarkPhoton/<E>.txt.gz, columns "<m_ap>(<config>)")
        # and writes the chi_2 tables in the same format (files/direct/iDM/
        # <E>.txt.gz, columns "<m_chi2>(<config>)").
        stem = energy_stem(energy)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filein = os.path.join(root, "files", "direct", "DarkPhoton", stem + ".txt.gz")
        fileout = os.path.join(root, "files", "direct", "iDM", stem + ".txt.gz")

        # read the A' column labels and group them by A' mass
        with gzip.open(filein, "rb") as f:
            header = f.readline().decode().split()[2:]
        groups = {}
        for col in header:
            mstr, config = col[:-1].split("(")
            groups.setdefault(mstr, []).append((config, col))

        # canonical (logth, logp) grid, used for masses with no surviving flux
        grid_th, grid_p, grid_zero = self.get_hist_list(np.array([]), np.array([]), np.array([]), prange)

        results = {}
        for mstr, items in groups.items():
            m0 = float(mstr)
            m1 = m0 / self.r
            m2 = m1 * (1 + self.delta)
            keys = [col for config, col in items]
            momenta, weights = self.read_list_4momenta_weights(filein, keys, mass=m0, nsample=nsample_load)
            print ('process:', stem, 'm_ap =', mstr)
            if len(momenta) == 0:
                cols = [grid_zero for _ in items]
            else:
                grid_th, grid_p, cols = self.get_X2_spectrum(momenta, weights, m0, m1, m2, nsample, prange)
            for (config, col), list_w in zip(items, cols):
                results[col] = (str(round(m2, 5)) + "(" + config + ")", list_w)

        # write the chi_2 spectra, preserving the A' column order
        out_keys = [results[col][0] for col in header]
        out_cols = [results[col][1] for col in header]
        os.makedirs(os.path.dirname(fileout), exist_ok=True)
        self.write_list_angle_momenta_weights(grid_th, grid_p, out_cols, out_keys, filename=fileout)


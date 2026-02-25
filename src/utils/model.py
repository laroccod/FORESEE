import numpy as np
from scipy import interpolate
from .utility import Utility

class Model(Utility):

    def __init__(self,name, path="./"):
        self.model_name = name
        self.dsigma_der_coupling_ref = None
        self.dsigma_der = None
        self.dEdx_coupling_ref= None
        self.dEdx = None
        self.recoil_max = "1e10"
        self.lifetime_coupling_ref = None
        self.lifetime_function = None
        self.br_mode=None
        self.br_functions = {}
        self.br_finalstate = {}
        self.production = {}
        self.modelpath = path

    ###############################
    #  Interaction Rate dsigma/dER
    ###############################

    def set_dsigma_drecoil_1d(self, dsigma_der, recoil_max="1e10", coupling_ref=1):
        """
        Set the cross section differential in recoil energy

        Parameters
        ----------
        dsigma_der: string
            Expression for the differential cross section d sigma / d E_r, with E_r the recoil energy
        recoil_max: string
            Expression for the maximum recoil value
        coupling_ref: float
            Reference coupling values. In most cases the xsec as a function of coupling g
            can be written as xsec(g) = xsec(g*) g*^2 / g^2 for some reference coupling g*

        Returns
        -------
            None
        """
        self.dsigma_der = dsigma_der
        self.dsigma_der_coupling_ref=coupling_ref
        self.recoil_max = recoil_max

    def set_dsigma_drecoil_2d(self, dsigma_der, recoil_max="1e10" ):
        """
        Set the cross section differential in recoil energy, w/o/ using/assuming a reference coupling

        Parameters
        ----------
        dsigma_der: string
            Expression for the differential cross section d sigma / d E_r, with E_r the recoil energy
        recoil_max: string
            Expression for the maximum recoil value

        Returns
        -------
            None
        """
        self.dsigma_der = dsigma_der
        self.dsigma_der_coupling_ref=None
        self.recoil_max = recoil_max

    def get_sigmaint(self, mass, coupling, energy, ermin, ermax):
        """
        Find interaction cross section

        Parameters
        ----------
        mass: float
            Particle mass, included implicitly in the expression self.dsigma_der
        coupling: float
            Coupling strength, included implicitly in the expression self.dsigma_der
        energy: float
            Incoming particle energy, included implicitly in the expression self.dsigma_der
        ermin: float
            Minimum particle energy
        ermax: float
            Maximum particle energy

        Returns
        -------
            None
        """
        minrecoil, maxrecoil = ermin, min(eval(self.recoil_max), ermax)
        nrecoil, sigma = 20, 0
        l10ermin, l10ermax = np.log10(minrecoil), np.log10(maxrecoil)
        dl10er = (l10ermax-l10ermin)/float(nrecoil)
        # df  = df / dx * dx = df/dx * dlog10x * x * log10
        for recoil in np.logspace(l10ermin+0.5*dl10er, l10ermax-0.5*dl10er, nrecoil):
            sigma += eval(self.dsigma_der) * recoil
        sigma *=  dl10er * np.log(10)
        return sigma

    def get_sigmaints(self, mass, couplings, energy, ermin, ermax):
        """
        Handle to different get_sigmaint use cases depending on dsigma_der and dsigma_der_coupling_ref values

        Parameters
        ----------
        mass: float
            Particle mass
        couplings: numpy array
            The couplings to scan over
        energy: float
            Incoming particle energy
        ermin: float
            Minimum particle energy
        ermax: float
            Maximum particle energy

        Returns
        -------
            Interaction cross sections for each coupling as a list of floats
        """
        if self.dsigma_der==None:
            print ("No interaction rate specified. You need to specify interaction rate first!")
            return 10**10
        elif self.dsigma_der_coupling_ref is None:
            sigmaints = [self.get_sigmaint(mass, coupling, energy, ermin, ermax) for coupling in couplings]
            return sigmaints
        else:
            sigmaint_ref = self.get_sigmaint(mass, self.dsigma_der_coupling_ref, energy, ermin, ermax)
            sigmaints = [ sigmaint_ref * coupling**2 / self.dsigma_der_coupling_ref**2  for coupling in couplings]
            return sigmaints
            
    ###############################
    #  Ionization
    ###############################
    
    def set_dEdx(self, dEdx, coupling_ref=1, K=0.307075):
        self.dEdx = dEdx
        self.dEdx_coupling_ref = coupling_ref
        self.K = K
        
    def get_dEdx_ref(self, mass, coupling, energy):
        K = self.K
        return eval(self.dEdx)
        
    def get_dEdx(self, mass, couplings, energy):
        K = self.K
        dEdx_ref = self.get_dEdx_ref(mass, self.dEdx_coupling_ref, energy)
        dEdx = [ dEdx_ref * coupling**2 / self.dEdx_coupling_ref**2  for coupling in couplings]
        return dEdx

    ###############################
    #  Lifetime
    ###############################

    def set_ctau_1d(self,filename, coupling_ref=1):
        """
        Set up ctau values read from model-specific input tables

        Parameters
        ----------
        filename: str
            The name of the file under modelpath to read ctau values from
        coupling_ref: float
            Reference coupling values

        Returns
        -------
            None
        """
        data=np.loadtxt(self.modelpath+filename).T
        self.ctau_coupling_ref=coupling_ref
        self.ctau_function=interpolate.interp1d(data[0], data[1],fill_value="extrapolate")

    def set_ctau_2d(self,filename):
        """
        Set up ctau values read from model-specific input tables, depending on mass and coupling (hence 2d)

        Parameters
        ----------
        filename: str
            The name of the file under modelpath to read ctau values from

        Returns
        -------
            None
        """
        data=np.loadtxt(self.modelpath+filename).T
        self.ctau_coupling_ref=None
        #try:
        #    self.ctau_function=interpolate.interp2d(data[0], data[1], data[2], kind="linear",fill_value="extrapolate")
        #except:
        nx = len(np.unique(data[0]))
        ny = int(len(data[0])/nx)
        self.ctau_function=interpolate.interp2d(data[0].reshape(nx,ny).T[0], data[1].reshape(nx,ny)[0], data[2].reshape(nx,ny).T, kind="linear",fill_value="extrapolate")

    def get_ctau(self,mass,coupling):
        if self.ctau_function==None:
            print ("No lifetime specified. You need to specify lifetime first!")
            return 10**10
        elif self.ctau_coupling_ref is None:
            return self.ctau_function(mass,coupling)[0]
        else:
            return self.ctau_function(mass) / coupling**2 *self.ctau_coupling_ref**2

    ###############################
    #  BR
    ###############################

    def set_br_1d(self,modes, filenames, finalstates=None):
        """
        Set up a decay modes via branching fractions.
        The 1D decay modes's br functions take mass as input argument.

        Parameters
        ----------
        modes: [str]
            List of strings indicating decay modes i.e. final state particles, e.g. ["e_e","mu_mu"]
        filenames: [str]
            List of strings indicating br table input filenames, w/ datatype suffix
        finalstates: [[int,int]] / [None]
            Table of PDG IDs corresponding to the final state particles of each decay mode

        Returns
        -------
            None
        """
        self.br_mode="1D"
        self.br_functions = {}
        if finalstates==None: finalstates=[None for _ in modes]
        for channel, filename, finalstate in zip(modes, filenames, finalstates):
            data = np.loadtxt(self.modelpath+filename).T
            function = interpolate.interp1d(data[0], data[1],fill_value="extrapolate")
            self.br_functions[channel] = function
            self.br_finalstate[channel] = finalstate

    def set_br_2d(self,modes,filenames, finalstates=None):
        """
        Set up a decay modes via branching fractions.
        The 2D decay modes's br functions take mass and coupling as input arguments.

        Parameters
        ----------
        modes: [str]
            List of strings indicating decay modes i.e. final state particles, e.g. ["e_e","mu_mu"]
        filenames: [str]
            List of strings indicating br table input filenames, w/ datatype suffix
        finalstates: [[int,int]] / [None]
            Table of PDG IDs corresponding to the final state particles of each decay mode

        Returns
        -------
            None
        """
        self.br_mode="2D"
        self.br_functions = {}
        if finalstates==None: finalstates=[None for _ in modes]
        for channel, filename, finalstate in zip(modes, filenames, finalstates):
            data = np.loadtxt(self.modelpath+filename).T
            #try:
            #    function = interpolate.interp2d(data[0], data[1], data[2], kind="linear",fill_value="extrapolate")
            #except:
            nx = len(np.unique(data[0]))
            ny = int(len(data[0])/nx)
            function = interpolate.interp2d(data[0].reshape(nx,ny).T[0], data[1].reshape(nx,ny)[0], data[2].reshape(nx,ny).T, kind="linear",fill_value="extrapolate")
            self.br_functions[channel] = function
            self.br_finalstate[channel] = finalstate

    def get_br(self,mode,mass,coupling=1):
        """
        Fetch the branching fraction functions stored into a class instance

        Parameters
        ----------
        mode: string / None
            Channel for which to fetch the branching fraction function
        mass: float
            The particle mass
        coupling: float
            Optional coupling strength value for 2D br_mode

        Returns
        -------
            Dictionary of branching fraction functions, with channel/mode strings as keys
        """
        if self.br_mode==None:
            print ("No branching fractions specified. You need to specify branching fractions first!")
            return 0
        elif mode not in self.br_functions.keys():
            print ("No branching fractions into ", mode, " specified. You need to specify BRs for this channel!")
            return 0
        elif self.br_mode == "1D":
            return self.br_functions[mode](mass)
        elif self.br_mode == "2D":
            return self.br_functions[mode](mass, coupling)[0]


    ###############################
    #  Production
    ###############################

    def add_production_2bodydecay(self, pid0, pid1, br, generator, energy, nsample_had=1, nsample=1, label=None, massrange=None, scaling=2, preselectioncut=None):
        """
        Introduce a 2-body decay production mode, from a SM initial state to a SM+exotic final state

        Parameters
        ----------
        pid0: string
            The PDG ID of the initial state particle
        pid1: string
            The PDG ID of the SM decay product
        br: str / types.FunctionType
            The expression to be computed
        generator: [str]
            List of predictions to consider, e.g. ['EPOSLHC', 'SIBYLL', ...]
        energy: str
            Collider sqrt(S) in TeV
        nsample_had: int
            Number of Monte Carlo samples to consider for mother hadrons,
            see nsample in read_list_4momenta_weights
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
        label: str / None
            Label for the production mode, serves as key for production dict.
            Default to initial state PDG ID if None
        massrange: [float,float]
            Lower and upper limit for masses to consider
        scaling: float / str
            If float, the scaling power if cross section at a given coupling estimated according
            to its ratio to a reference coupling, with the ratio raised to the scaling power.
            Alternatively e.g. "manual", see get_production_scaling
        preselectioncuts: str / None
            Expression defining cuts to be used e.g. "th<0.01 and p>100"

        Returns
        -------
            None
        """
        if label is None: label=pid0
        if type(generator)==str: generator=[generator]
        if type(br       )==str: br=br.replace("'pid0'","'"+str(pid0)+"'").replace("'pid1'","'"+str(pid1)+"'")
        self.production[label]= {"type": "2body", "pid0": pid0, "pid1": pid1, "pid2": None, "br": br, "production": generator, "energy": energy, "nsample_had": nsample_had, "nsample": nsample, "massrange": massrange, "scaling": scaling, "preselectioncut": preselectioncut, "integration": None}


    def add_production_3bodydecay(self, pid0, pid1, pid2, br, generator, energy, nsample_had=1, nsample=1, label=None, massrange=None, scaling=2, preselectioncut=None, integration="dq2dcosth"):
        """
        Introduce a 3-body decay production mode, where a SM initial state decays into a SM particle + 2 exotic particles of the same kind

        Parameters
        ----------
        pid0: string
            The PDG ID of the initial state particle
        pid1: string
            The PDG ID of the SM decay product
        pid2: string
            The PDG ID of the other decay product, "0" for exotic
        br: str / types.FunctionType
            The expression to be computed
        generator: [str]
            List of predictions to consider, e.g. ['EPOSLHC', 'SIBYLL', ...]
        energy: str
            Collider sqrt(S) in TeV
        nsample_had: int
            Number of Monte Carlo samples to consider for mother hadrons,
            see nsample in read_list_4momenta_weights
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
        label: str / None
            Label for the production mode, serves as key for production dict.
            Default to initial state PDG ID if None
        massrange: [float,float]
            Lower and upper limit for masses to consider
        scaling: float / str
            If float, the scaling power if cross section at a given coupling estimated according
            to its ratio to a reference coupling, with the ratio raised to the scaling power.
            Alternatively e.g. "manual", see get_production_scaling
        preselectioncuts: str, None
            Expression defining cuts to be used e.g. "th<0.01 and p>100"

        Returns
        -------
            None
        """
        if label is None: label=pid0
        if type(generator)==str: generator=[generator]
        if type(br       )==str: br=br.replace("'pid0'","'"+str(pid0)+"'").replace("'pid1'","'"+str(pid1)+"'").replace("'pid2'","'"+str(pid2)+"'")
        self.production[label]= {"type": "3body", "pid0": pid0, "pid1": pid1, "pid2": pid2, "br": br, "production": generator, "energy": energy, "nsample_had": nsample_had, "nsample": nsample, "massrange": massrange, "scaling": scaling, "preselectioncut": preselectioncut, "integration": integration}

    def add_production_mixing(self, pid, mixing, generator, energy, label=None, massrange=None, scaling=2):
        """
        Introduce mixing as a production mode

        Parameters
        ----------
        pid: string
            The PDG ID of the particle with which the mixing occurs
        mixing: str / types.FunctionType
            The expression to be computed
        generator: [str]
            List of predictions to consider, e.g. ['EPOSLHC', 'SIBYLL', ...]
        energy: str
            Collider sqrt(S) in TeV
        label: str / None
            Label for the production mode, serves as key for production dict.
            Default to PDG ID if None
        massrange: [float,float]
            Lower and upper limit for masses to consider
        scaling: float / str
            If float, the scaling power if cross section at a given coupling estimated according
            to its ratio to a reference coupling, with the ratio raised to the scaling power.
            Alternatively e.g. "manual", see get_production_scaling

        Returns
        -------
            None
        """
        if label is None: label=pid
        if type(generator)==str: generator=[generator]
        if type(mixing   )==str: mixing=mixing.replace("'pid'","'"+str(pid)+"'")
        self.production[label]= {"type": "mixing", "pid0": pid, "mixing": mixing, "production": generator, "energy": energy, "massrange": massrange, "scaling": scaling}

    def add_production_direct(self, label, energy, configuration=None, coupling_ref=1, condition="True", masses=None, scaling=2):
        """
        Introduce a mode of direct production

        Parameters
        ----------
        label: str
            Label for the production mode, e.g. "Brem". Serves as key for production dict.
            Expect to find model-specific tables under model/direct/*/label_*.txt
        energy: str
            Collider sqrt(S) in TeV
        coupling_ref: float
            Reference coupling value
        condition: str / [str]
            The condition specifying the production
        masses: [float]
            Particle masses to consider
        scaling: float / str
            If float, the scaling power if cross section at a given coupling estimated according
            to its ratio to a reference coupling, with the ratio raised to the scaling power.
            Alternatively e.g. "manual", see get_production_scaling

        Returns
        -------
            None
        """
        if condition == None: condition='1'
        if type(condition) in [float, int]: condition=str(condition)
        if type(condition) == str: condition=[condition]
        if configuration == None: configuration=label
        if type(configuration)==str: configuration=[configuration]
        if (len(configuration)>1) and (len(condition)>1): print ("You can only have multiple conditions OR multiple configurations!")
        production = condition if len(condition)>1 else configuration
        self.production[label]= {"type": "direct", "energy": energy, "masses": masses, "scaling": scaling, "coupling_ref": coupling_ref, "production": production, "configuration": configuration, "condition": condition}

    def get_production_scaling(self, key, mass, coupling, coupling_ref):
        """
        Scaling factor for estimating a coupling-dependent quantity based on its ratio to a
        reference coupling, raised to some scaling power

        Parameters
        ----------
        key: str
            The production dictionary key corresponding to this mode
        mass: float
            Particle mass, included implicitly via eval statements
        coupling: float
            The coupling value at which to estimate the result
        coupling_ref: float
            Reference coupling value

        Returns
        -------
            The resulting factor as a float
        """
        scaling = self.production[key]["scaling"]
        if self.production[key]["type"] in ["2body","3body"]:
            if scaling == "manual":
                return eval(self.production[key]["br"], {"self":self, "np":np, "mass":mass, "coupling":coupling})/eval(self.production[key]["br"], {"self":self, "np":np, "mass":mass, "coupling":coupling_ref})
            else: return (coupling/coupling_ref)**scaling
        if self.production[key]["type"] == "mixing":
            if scaling == "manual":
                return eval(self.production[key]["mixing"], {"coupling":coupling})**2/eval(self.production[key]["mixing"], {"coupling":coupling_ref})**2
            else: return (coupling/coupling_ref)**scaling
        if self.production[key]["type"] == "direct":
            return (coupling/coupling_ref)**scaling

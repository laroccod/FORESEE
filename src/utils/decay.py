from .vectors import *

class Decay():

    def __init__(self, rng=None):
        self.rng = rng

    ###############################
    #  Kinematic Functions
    ###############################

    def twobody_decay(self, p0, m0, m1, m2, phi, costheta):
        """
        Function that decays p0 -> p1 p2 and returns p1,p2

        Note that if more than one of m0,m1,m2,phi,costheta are given as arrays, the dimensions of any 
        two arrays must agree. The manipulations of energies and momenta below are designed to work
        with numpy array syntax, which performs operations element-by-element.

        Parameters
        ----------
        p0: LorentzVector / [LorentzVector] / skheparray
            Initial state particle 4-momentum
        m0: float / np.array(float)
            Mass of the incoming particle
        m1: float / np.array(float)
            First final state particle mass
        m2: float / np.array(float)
            Second final state particle mass
        phi: float / np.array(float)
            Azimuthal angle
            Must be within (-pi, pi)
        costheta: float / np.array(float)
            Cosine of the polar angle
            Must be within (-1., 1.)
        
        Returns
        -------
            Boosted p1,p2 as LorentzVectors if phi & costheta are floats, or if input variables 
            are arrays, return a list of LorentzVectors (old skhep) / skheparray (new skhep)
        """
                
        #check if parameters given as float or arrays/lists
        arr_pars = sum([type(var) in [list,np.ndarray] for var in[m0,m1,m2,phi,costheta]])>0
        arrlen = max(list(map(lambda var: len(np.array([var]).ravel()), [costheta,phi,m0,m1,m2])))
                
        #energy and momentum of p2 in the rest frame of p0
        #N.B. if any m are arrays, both energy1,2 & momentum1,2 will also be, and so also px,py,pz,en
        energy1   = (m0*m0+m1*m1-m2*m2)/(2.*m0)
        energy2   = (m0*m0-m1*m1+m2*m2)/(2.*m0)
        momentum1 = np.sqrt(energy1*energy1-m1*m1)
        momentum2 = np.sqrt(energy2*energy2-m2*m2)

        #for uniform internal handling in case m aren't arrays, turn angles into arrays s.t. at
        #least one term in px,py,pz,en expressions below is an np.array, making px,py,pz,en np.arrays
        costheta_arr = np.array([costheta]).ravel()
        phi_arr = np.array([phi]).ravel() 

        #4-momenta of p1 and p2 in the rest frame of p0
        en1 = energy1*np.ones(arrlen)
        pz1 = momentum1 * costheta_arr
        py1 = momentum1 * np.sqrt(1.-np.power(costheta_arr,2)) * np.sin(phi_arr)
        px1 = momentum1 * np.sqrt(1.-np.power(costheta_arr,2)) * np.cos(phi_arr)
        p1 = LorentzArray({'px':-px1,'py':-py1,'pz':-pz1,'energy':en1})        
        en2 = energy2*np.ones(arrlen)
        pz2 = momentum2 * costheta_arr
        py2 = momentum2 * np.sqrt(1.-np.power(costheta_arr,2)) * np.sin(phi_arr)
        px2 = momentum2 * np.sqrt(1.-np.power(costheta_arr,2)) * np.cos(phi_arr)
        p2 = LorentzArray({'px':px2,'py':py2,'pz':pz2,'energy':en2})
        
        #check if an array of initial state particle momenta was given, or a single momentum
        if type(p0)==LorentzVector: arr_p0 = False
        else:
            try:
                if type(p0[0].px) in [np.float64,float]: arr_p0 = True
                p0len = len(p0)
                if arr_pars and p0len!=arrlen:
                    print('WARNING twobody_decay p0 and other parameter array length mismatch',p0len,arrlen) 
            except: arr_p0 = False
        
        #assert initial state particle momenta/momentum datatype
        #If both p0 & other pars are arrays, dim must agree: each par corresponds to a p0 of it's own
        p0_ = p0
        if not arr_p0 and type(p0_)!=LorentzVector: p0_ = LorentzVector(px=p0.x,py=p0.y,pz=p0.z,e=p0.e)
        
        #get angle and axis of p0
        zaxis=Vector3D(0,0,1)
        rotangle = get_rotangle(refax=zaxis, p4=p0_)
        rotaxis  = get_rotaxis( refax=zaxis, p4=p0_)
        
        #Rotate
        p1 = rotateLorentzArray(momenta=p1,rotangle=rotangle,rotaxis=rotaxis)
        p2 = rotateLorentzArray(momenta=p2,rotangle=rotangle,rotaxis=rotaxis)
        
        #boost in p0 restframe
        p1_ = boostLorentzArray(momenta=p1,boostby=p0_,boostfactor=-1.)
        p2_ = boostLorentzArray(momenta=p2,boostby=p0_,boostfactor=-1.)

        #Return array or LorentzVectors depending on input angles's datatypes
        if arr_pars or arr_p0: return p1_,p2_
        else: return LorentzVector(p1_[0].px,p1_[0].py,p1_[0].pz,p1_[0].e),\
                     LorentzVector(p2_[0].px,p2_[0].py,p2_[0].pz,p2_[0].e)

    #TODO can probably be streamlined similar to the other threebody decay functions
    def threebody_decay_pure_phase_space(self, p0, m0, m1, m2, m3):
        """
        Function that decays p0 > p1 p2 p2 and returns p1,p2,p3
        following pure phase space

        Parameters
        ----------
        p0: LorentzVector
            Initial state particle 4-momentum
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        Returns
        -------
            Boosted p1,p2,p3 as LorentzVectors
        """

        p1, p2, p3 = None, None, None
        m1p2 = (m1+m2)**2
        m0m3 = (m0-m3)**2
        m2p3 = (m2+m3)**2
        m0m1 = (m0-m1)**2
        while p1 == None:
            #randomly draw mij^2
            m122 = self.rng.uniform(m1p2, m0m3)
            m232 = self.rng.uniform(m2p3, m0m1)
            m132 = m0**2+m1**2+m2**2+m3**2-m122-m232

            #calculate energy and momenta
            e1 = (m0**2+m1**2-m232)/(2*m0)
            e2 = (m0**2+m2**2-m132)/(2*m0)
            e3 = (m0**2+m3**2-m122)/(2*m0)
            
            #Make sure unphysical energies not sampled. Suffices to check e2 since
            #e1>m1 <=> m0**2+m1**2-m232 > 2*m0*m1 <=> m232 < (m0-m1)**2,
            #e3>m3 <=> m0**2+m3**2-m122 > 2*m0*m3 <=> m122 < (m0-m3)**2
            #hold by definition
            if e2<m2: continue
            
            #3-momentum magnitudes
            mom1 = np.sqrt(e1**2-m1**2)
            mom2 = np.sqrt(e2**2-m2**2)
            mom3 = np.sqrt(e3**2-m3**2)

            #calculate angles
            costh12 = (-m122 + m1**2 + m2**2 + 2*e1*e2)/(2*mom1*mom2)
            costh13 = (-m132 + m1**2 + m3**2 + 2*e1*e3)/(2*mom1*mom3)
            costh23 = (-m232 + m2**2 + m3**2 + 2*e2*e3)/(2*mom2*mom3)
            if (abs(costh12)>1) or (abs(costh13)>1) or (abs(costh23)>1): continue

            sinth12 =  np.sqrt(1-costh12**2)
            sinth13 =  np.sqrt(1-costh13**2)
            sinth23 =  np.sqrt(1-costh23**2)

            #construct momenta
            p1 = LorentzVector(mom1,0,0,e1)
            p2 = LorentzVector(mom2*costh12, mom2*sinth12,0,e2)
            p3 = LorentzVector(mom3*costh13,-mom3*sinth13,0,e3)
            break

        #random rotation of p2, p3 around p1
        xaxis=Vector3D(1,0,0)
        phi = self.rng.uniform(-math.pi,math.pi)
        p1=p1.rotate(phi,xaxis)
        p2=p2.rotate(phi,xaxis)
        p3=p3.rotate(phi,xaxis)

        #random rotation of p1 in ref frame
        phi = self.rng.uniform(-math.pi,math.pi)
        costh = self.rng.uniform(-1,1)
        theta = np.arccos(costh)
        axis=Vector3D(np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta))
        rotaxis=axis.cross(p1.vector).unit()
        rotangle=axis.angle(p1.vector)
        p1=p1.rotate(rotangle,rotaxis)
        p2=p2.rotate(rotangle,rotaxis)
        p3=p3.rotate(rotangle,rotaxis)

        #boost in p0 restframe
        p1_=p1.boost(-1.*p0.boostvector)
        p2_=p2.boost(-1.*p0.boostvector)
        p3_=p3.boost(-1.*p0.boostvector)
        
        return p1_, p2_, p3_

    ###############################
    #  sample hadron decays n times
    ###############################

    def decay_in_restframe_2body(self, br, m0, m1, m2, nsample):
        """
        Handle to call the twobody_decay function repeatedly to form a MC sample

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            List of particle 4-momenta, list of weights resulting from branching fraction divided by MC sample size
        """
        # prepare output
        particles = []
        weights = br/nsample*np.ones(nsample)

        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #Sample random angles
        #cos = np.array(list(map(self.rng.uniform,     [-1.]*nsample,     [1.]*nsample)))
        #phi = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)))
        #TODO replace below w/ above: more efficient but fail tests as generation ordering differs
        cos,phi=[],[]
        for i in range(nsample):
            cos.append(self.rng.uniform(-1.,1.))
            phi.append(self.rng.uniform(-math.pi,math.pi))
        
        _,particles = self.twobody_decay(p_mother,m0,m1,m2,phi,cos)
        
        return particles,weights

    def decay_in_restframe_3body(self, br, coupling, m0, m1, m2, m3, nsample, integration):
        """
        Handle to the various integration options of 3-body decays

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        coupling: float
            Coupling strength
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by
        integration: str
            Specifies which 3body decay function to call

        Returns
        -------
            List of LLP momenta as LorentzVectors and a list of weights
        """

        if integration == "dq2dcosth":
            return self.decay_in_restframe_3body_dq2dcosth(br, coupling, m0, m1, m2, m3, nsample)
        if integration == "dq2dE":
            return self.decay_in_restframe_3body_dq2dE(br, coupling, m0, m1, m2, m3, nsample)
        if integration == "dE":
            return self.decay_in_restframe_3body_dE(br, coupling, m0, m1, m2, m3, nsample)
        if integration == "chain_decay":
            mass = m3
            mI = eval(br[1])
            if (m0 <= m1+mI) or (mI<m2+m3): return [LorentzVector(0,0,0,m0)], [0]
            return self.decay_in_restframe_3body_chain(eval(br[0]), coupling, m0, m1, m2, m3, mI, nsample)

    def decay_in_restframe_3body_dq2dcosth(self,br, coupling, m0, m1, m2, m3, nsample):
        """
        3-body decay function with integration over q^2 and cos(theta), with theta the angle to z-axis

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        coupling: float
            Coupling strength, included implicitly via eval(br)
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            List of particle 4-momenta, list of weights resulting from branching fraction divided by MC sample size
        """

        #create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        #integration boundary
        q2min,q2max = (m2+m3)**2,(m0-m1)**2
        cthmin,cthmax = -1. , 1.
        mass = m3

        #Sample kinematic variables
        #q2   = np.array(list(map(self.rng.uniform,[q2min   ]*nsample,[q2max  ]*nsample)))
        #cth  = np.array(list(map(self.rng.uniform,[cthmin  ]*nsample,[cthmax ]*nsample)))
        #phiQ = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)))
        #cosM = np.array(list(map(self.rng.uniform,[-1.     ]*nsample,[1.     ]*nsample)))
        #phiM = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)))
        #TODO replace below w/ above: more efficient but generation order differs
        q2,cth,phiQ,cosM,phiM = [],[],[],[],[]
        for i in range(nsample):
            q2  .append(self.rng.uniform(q2min,   q2max  ))
            cth .append(self.rng.uniform(cthmin,  cthmax ))
            phiQ.append(self.rng.uniform(-math.pi,math.pi))
            cosM.append(self.rng.uniform(-1.,     1.     ))
            phiM.append(self.rng.uniform(-math.pi,math.pi))
        
        #Redefinitions
        th = np.arccos(cth)  #Definition required for eval(br) below
        q  = np.sqrt(q2)
        cosQ = cth
            
        #Decay meson and V
        particles=[]
        p_1,p_q = self.twobody_decay(p_mother,m0,m1,q,phiM,cosM)
        _,particles = self.twobody_decay(p_q,q,m2,m3,phiQ,cosQ)
            
        #branching fraction
        brval = eval(br)
        brval *= (q2max-q2min)*(cthmax-cthmin)/float(nsample)
        weights = brval*np.ones(nsample)

        return particles,weights

    def decay_in_restframe_3body_dq2dE(self, br, coupling, m0, m1, m2, m3, nsample):
        """
        3-body decay function with integration over q^2 and energy

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        coupling: float
            Coupling strength, included implicitly via eval(br)
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            List of particle 4-momenta, list of weights resulting from branching fraction divided by MC sample size
        """

        # prepare output
        particles = []

        #integration boundary
        q2min,q2max = (m2+m3)**2,(m0-m1)**2
        mass = m3

        #sample random variables
        #q2    = np.array(list(map(self.rng.uniform,[   q2min]*nsample,[  q2max]*nsample)))
        #costh = np.array(list(map(self.rng.uniform,[     -1.]*nsample,[     1.]*nsample)))
        #phi   = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)))
        #TODO replace below w/ above, more efficient but generation order differs
        q2,costh,phi,energy=[],[],[],[]
        for _ in range(nsample):
            #FIXME should use self.rng.uniform instead of random.uniform --> agree
            q2.append(random.uniform(q2min,q2max))
            costh.append(random.uniform(-1,1))
            phi.append(random.uniform(-math.pi,math.pi))
            energy.append(random.uniform(0.,1.))  #Translated to ENmin, ENmax interval below
        #Ensure np.array format
        q2 = np.array(q2)
        costh = np.array(costh)
        phi = np.array(phi)
        
        q = np.sqrt(q2)
        sinth = np.sqrt(1-costh**2)
        
        E2st = (q**2 - m2**2 + m3**2)/(2*q)
        E3st = (m0**2 - q**2 - m1**2)/(2*q)
        m232min = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) + np.sqrt(E3st**2 - m1**2))**2
        m232max = (E2st + E3st)**2 - (np.sqrt(E2st**2 - m3**2) - np.sqrt(E3st**2 - m1**2))**2
        cthmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        cthmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
        ENmax = (m232max + q**2 - m2**2 - m1**2)/(2*m0)
        ENmin = (m232min + q**2 - m2**2 - m1**2)/(2*m0)
        
        #energy = np.array(list(map(self.rng.uniform,ENmin,ENmax)))
        #TODO replace w/ above, then also remove energy from the _ in range(nsample) loop above
        energy = energy*(ENmax-ENmin) + ENmin

        # get LLP momenta
        p = np.sqrt(energy**2-mass**2)
        particles = LorentzArray({'px': p*sinth*np.cos(phi),\
                                  'py': p*sinth*np.sin(phi),\
                                  'pz': p*costh,\
                                  'energy':energy})
        
        #branching fraction
        brval  = eval(br)
        brval *= (q2max-q2min)*(ENmax-ENmin)/float(nsample)
        weights = brval*np.ones(nsample)

        return particles,weights

    def decay_in_restframe_3body_dE(self, br, coupling, m0, m1, m2, m3, nsample):
        """
        3-body decay function with integration over energy

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        coupling: float
            Coupling strength, included implicitly via eval(br)
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            List of particle 4-momenta, list of weights resulting from branching fraction divided by MC sample size
        """

        # prepare output
        particles, weights = [], []
        mass = m3

        #integration boundary
        emin, emax = m3, (m0**2+m3**2-(m1+m2)**2)/(2*m0)

        #numerical integration
        integral=0  #TODO is "integral" actually used anywhere?
        
        #energy = np.array(list(map(self.rng.uniform,[    emin]*nsample,[   emax]*nsample)))
        #phi    = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)))
        #costh  = np.array(list(map(self.rng.uniform,[     -1.]*nsample,[     1.]*nsample)))
        #TODO replace below w/ above
        energy,phi,costh = [],[],[]
        for i in range(nsample):
            energy.append(random.uniform(emin,emax))
            phi.append(random.uniform(-math.pi,math.pi))  #FIXME self.rng.uniform
            costh.append(random.uniform(-1,1))  #FIXME self.rng.uniform
        #Ensure array format
        energy = np.array(energy)
        costh = np.array(costh)
        phi = np.array(phi)
        
        # get LLP momenta
        sinth = np.sqrt(1-costh**2)        
        p = np.sqrt(energy**2-mass**2)        
        particles = LorentzArray({'px': p*sinth*np.cos(phi),\
                                  'py': p*sinth*np.sin(phi),\
                                  'pz': p*costh,\
                                  'energy': energy})
        
        #branching fraction
        brval  = eval(br)
        brval *= (emax-emin)/float(nsample)
        weights = brval*np.ones(nsample)

        return(particles, weights)

    def decay_in_restframe_3body_chain(self, br, coupling, m0, m1, m2, m3, mI, nsample):
        """
        3-body decay as a chain of 2-body decays: m0 -> m1 mI, mI -> m2 m3

        Parameters
        ----------
        br: str / types.FunctionType
            Branching fraction function for the considered mode
        coupling: float
            Coupling strength, included implicitly via br
        m0: float
            Initial state particle mass
        m1: float
            First final state particle mass
        m2: float
            Second final state particle mass
        m3: float
            Third final state particle mass
        mI: float
            The intermediate particle mass, from which m2 and m3 are produced
        nsample: int
            Number of Monte Carlo samples to add into particles, and to divide weights by

        Returns
        -------
            List of particle 4-momenta, list of weights resulting from branching fraction divided by MC sample size
        """

        # prepare output
        particles = []

        # create parent 4-vector
        p_mother=LorentzVector(0,0,0,m0)

        # sample kinematic variables
        #cosI = np.array(list(map(self.rng.uniform,[     -1.]*nsample,[     1.]*nsample)
        #phiI = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)
        #cosM = np.array(list(map(self.rng.uniform,[     -1.]*nsample,[     1.]*nsample)
        #phiM = np.array(list(map(self.rng.uniform,[-math.pi]*nsample,[math.pi]*nsample)
        #TODO replace below w/ above
        cosI,phiI,cosM,phiM=[],[],[],[]
        for i in range(nsample):
            #FIXME self.rng.random
            cosI.append(random.uniform(-1.,1.))
            phiI.append(random.uniform(-math.pi,math.pi))
            cosM.append(random.uniform(-1.,1.))
            phiM.append(random.uniform(-math.pi,math.pi))

        # numerical integration TODO integration? Is integral actually used anywhere?
        p_1,p_I=self.twobody_decay(p_mother, m0 ,m1, mI ,phiM, cosM)
        _,particles=self.twobody_decay(p_I, mI ,m2, m3 ,phiI, cosI)

        #branching fraction
        brval = br/float(nsample)
        weights = brval*np.ones(nsample)

        return particles,weights

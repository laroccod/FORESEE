import numpy as np
from src.foresee import Utility, Foresee
import sympy as smp
import mpmath as mp
from os import listdir
import glob
import os
from os.path import exists
from HNL_Decay import *

class HeavyNeutralLepton(Utility):

    ###############################
    #  Initiate
    ###############################

    def __init__(self, ve=1, vmu=0, vtau=0):
        self.vcoupling = {"11": ve, "13":vmu, "15": vtau}   #HNL coupling to the electron, the muon and the tau lepton
        self.lepton = {"11": "e", "13":"mu", "15": "tau"}
        self.hadron = {"211": "pi", "321": "K", "213": "rho"}
    
    #decay constants
    def fH(self,pid):
        if   pid in ["211","-211","111"]: return 0.1303
        elif pid in ["221","-221"]: return 0.0784   
        elif pid in ["213","-213"]: return 0.210        #https://iopscience.iop.org/article/10.1088/1674-1137/42/7/073102/pdf
        elif pid in ["113"]       : return 0.220       #https://arxiv.org/pdf/1005.1607.pdf
        elif pid in ["223","-223"]: return 0.195    #https://arxiv.org/pdf/1005.1607.pdf
        elif pid in ["333"]: return 0.241               #for phi meson found here https://iopscience.iop.org/article/10.1088/1674-1137/abcd8f/pdf
        elif pid in ["313","-313","323","-323"]: return 0.204     
        elif pid in ["311","-311","321","-321"]: return 0.1564 
        elif pid in ["331","-331"]: return -0.0957
        elif pid in ["421","-421","411","-411"]: return 0.2226 
        elif pid in ["443"]: return 0.409         #J/psifound fromhttps://arxiv.org/pdf/hep-ph/9703252.pdf
        elif pid in ["431","-431"]: return 0.2801
        elif pid in ["511", "511","521","-521"]: return 0.190
        elif pid in ["513","-513"]: return 1.027*0.190
        elif pid in ["531","-531"]: return 0.230  
        elif pid in ["523","-523"]: return 1.027*0.190
        elif pid in ["533","-533"]: return 1.028*0.230        #obtained from https://iopscience.iop.org/article/10.1088/1674-1137/42/7/073102/pdf (same with all vector mesons)
        elif pid in ["541","-541"]: return 0.480
        elif pid in ["413","-413","423","-423"]: return 1.097*0.2226 
        elif pid in ["433","-433"]: return 1.093*0.2801

    # Lifetimes
    def tau(self,pid):
        if   pid in ["2112","-2112"]: return 10**8
        elif pid in ["15","-15"    ]: return 290.3*1e-15
        elif pid in ["2212","-2212"]: return 10**8
        elif pid in ["211","-211"  ]: return 2.603*10**-8
        #elif pid in ["223"         ]: return 7.58*10**-23 #obtained this lifetime from wiki, couldnt find on pdg
        elif pid in ["323","-323"  ]: return 1.425*10**-23 
        elif pid in ["321","-321"  ]: return 1.2380*10**-8
        elif pid in ["411","-411"  ]: return 1040*10**-15
        elif pid in ["421","-421"  ]: return 410*10**-15
        elif pid in ["423", "-423" ]: return 3.1*10**-22
        elif pid in ["431", "-431" ]: return 504*10**-15
        elif pid in ["511", "-511" ]: return 1.519*10**-12
        elif pid in ["521", "-521" ]: return 1.638*10**-12
        elif pid in ["531", "-531" ]: return 1.515*10**-12
        elif pid in ["541", "-541" ]: return 0.507*10**-12
        elif pid in ["310" ,"-310"        ]: return 8.954*10**-11
        elif pid in ["130" ,"-130"        ]: return 5.116*10**-8
        elif pid in ["3122","-3122"]: return 2.60*10**-10
        elif pid in ["3222","-3222"]: return 8.018*10**-11
        elif pid in ["3112","-3112"]: return 1.479*10**-10
        elif pid in ["3322","-3322"]: return 2.90*10**-10
        elif pid in ["3312","-3312"]: return 1.639*10**-10
        elif pid in ["3334","-3334"]: return 8.21*10**-11

    # CKM matrix elements
    def VH(self,pid):
        if   pid in ["211","-211"]: return 0.97373 #Vud
        elif pid in ["321","-321","323","-323"]: return 0.2243 #Vus
        elif pid in ["213","-213"]: return 0.97373
        elif pid in ["411","-411"]: return 0.221
        elif pid in ["431","-431","433","-433"]: return 0.975 #Vcs
        elif pid in ["541","-541"]: return 40.8E-3
        elif pid in ["411","-411","413","-413"]: return 0.221 #Vcd
        elif pid in ["541","-541"]: return 40.8E-3 #Vcb
        elif pid in ["521","-521","523","-523"]: return 3.82E-3 #Vub
        elif pid in []: return 8.6E-3 #Vtd
        elif pid in []: return 41.5E-3 #Vts
        elif pid in []: return 1.014 #Vtb

    #symbol for a given pid
    #originally created to analyze HNL decays
    def symbols(self,pid):
        #quarks
        if   pid in ["1"]: return "d"
        elif pid in ["2"]: return "u"
        elif pid in ["3"]: return "s"
        elif pid in ["4"]: return "c"
        elif pid in ["5"]: return "b"
        elif pid in ["6"]: return "t"

        #leptons
        elif pid in ["11"]: return "e"
        elif pid in ["12"]: return r"$\nu_e$"
        elif pid in ["13"]: return r"$\mu$"
        elif pid in ["14"]: return r"$\mu_e$"
        elif pid in ["15"]: return r"$\tau$"
        elif pid in ["16"]: return r"$\nu_{\tau}$"

        #neutral pseudoscalars
        elif pid in ["111"]: return r"$\pi^0$"
        elif pid in ["221"]: return r"$\eta$"
        elif pid in ["311"]: return r"$K^0$"
        elif pid in ["331"]: return r"$\eta^{'}$"
        elif pid in ["421"]: return r"$D^0$"

        #charged pseudoscalars
        elif pid in ["211"]: return r"$\pi^+$"
        elif pid in ["321"]: return r"$K^+$"
        elif pid in ["411"]: return r"$D^+$"
        elif pid in ["431"]: return r"$D^+_s$"
        elif pid in ["521"]: return r"$B^+$"

        #neutral vectors
        elif pid in ["113"]: return r"$\rho^0$"
        elif pid in ["223"]: return r"$\omega$"
        elif pid in ["313"]: return r"$K^{*0}$"
        elif pid in ["333"]: return r"$\phi$"
        elif pid in ["423"]: return r"$D^{*0}$"
        elif pid in ["443"]: return r"$J/\psi$"

        #charged vectors
        elif pid in ["213"]: return r"$\rho^+$"
        elif pid in ["323"]: return r"$K^{*+}$"
        elif pid in ["413"]: return r"$D^{*+}$"
        elif pid in ["433"]: return r"$D^{*+}_s$"




    #for HNL decays to neutral vector mesons
    def kV(self,pid):
        xw=0.23121
        if pid in ["313","-313"]: return (-1/4+(1/3)*xw)
        elif pid in ["423","-423","443"]: return (1/4-(2/3)*xw)
    
    def GF(self):
        return 1.1663788*10**(-5)

    ###############################
    #  2-body decays
    ###############################
    # Branching fraction
    #pid0 is parent meson, pid1 is daughter meson
    def get_2body_br(self,pid0,pid1):

        #read constant
        mH, mLep, tauH = self.masses(pid0), self.masses(pid1), self.tau(pid0)
        vH, fH = self.VH(pid0), self.fH(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)

        #calculate br
        prefactor=(tauH*GF**2*fH**2*vH**2)/(8*np.pi)
        prefactor*=self.vcoupling[str(abs(int(pid1)))]**2
        br=str(prefactor)+"*coupling**2*mass**2*"+str(mH)+"*(1.-(mass/"+str(mH)+")**2 + 2.*("+str(mLep)+"/"+str(mH)+")**2 + ("+str(mLep)+"/mass)**2*(1.-("+str(mLep)+"/"+str(mH)+")**2)) * np.sqrt((1.+(mass/"+str(mH)+")**2 - ("+str(mLep)+"/"+str(mH)+")**2)**2-4.*(mass/"+str(mH)+")**2)"
        return br

    #pid0 is tau lepton, pid1 is produced meson, pid2 is HNL
    def get_2body_br_tau(self,pid0,pid1):
        #for daugter vector meson rho added K^*; not sure if this is accurate
        if pid1 in ['213','-213','323','-323']:
            if pid1 in ['213','-213']: grho = 0.102
            if pid1 in ['323', '-323']: grho = 0.217*self.masses(pid1)
            VH, tautau = self.VH(pid1), self.tau(pid0)
            Mtau, Mrho=self.masses(pid0), self.masses(pid1)
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=(tautau*grho**2*GF**2*VH**2*Mtau**3/(8*np.pi*Mrho**2))
            prefactor*=(self.vcoupling[str(abs(int(pid0)))]**2)
            br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2+(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+((mass**2-2*self.masses('{pid1}')**2)/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2))*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2)))"
        #for daughter pseudoscalars
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)
            tautau=tautau*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            VH=self.VH(pid1)
            fH=self.fH(pid1)
            Mtau=self.masses(pid0)
            prefactor=(tautau*GF**2*VH**2*fH**2*Mtau**3/(16*np.pi))
            prefactor*=(self.vcoupling[str(abs(int(pid0)))]**2)
            br=f"{prefactor}*coupling**2*((1-(mass**2/self.masses('{pid0}')**2))**2-(self.masses('{pid1}')**2/self.masses('{pid0}')**2)*(1+(mass**2/self.masses('{pid0}')**2)))*np.sqrt((1-((self.masses('{pid1}')-mass)**2/self.masses('{pid0}')**2)*(1-((self.masses('{pid1}')+mass)**2/self.masses('{pid0}')**2))))"
        return (br)

    ###############################
    #  3-body decays
    ###############################

    #VHH in 3-body decays - CKM matrix elements
    def VHHp(self,pid0,pid1):     
        Vud=0.97373
        Vus=0.2243
        Vub=3.82E-3
        Vcd=0.221
        Vcs=0.975
        Vcb=40.8E-3
        Vtd=8.6E-3
        Vts=41.5E-3
        Vtb=1.014
        V21=Vud
        V23=Vus
        V25=Vub
        V41=Vcd
        V43=Vcs
        V45=Vcb

        if   pid0 in ["2","-2"    ] and pid1 in ["1","-1"    ]: return 0.97373 #Vud
        if   pid0 in ["2","-2","3","-3"] and pid1 in ["3","-3","2","-2"    ]: return 0.2243 #Vus
        if   pid0 in ["4","-4","1","-1"    ] and pid1 in ["1","-1","4","-4"   ]: return 0.221 #Vcd
        if   pid0 in ["4","-4","3","-3"    ] and pid1 in ["3","-3" ,"4","-4"   ]: return 0.975 #Vcs
        if   pid0 in ["4","-4","5","-5"    ] and pid1 in ["4","-4","5","-5"    ]: return 40.8E-3 #Vcb
        if   pid0 in ["2","-2","5","-5"    ] and pid1 in ["2","-2","5","-5"    ]: return 3.82E-3 #Vub
        if   pid0 in ["6","-6","1","-1"    ] and pid1 in ["6","-6","1","-1"    ]: return 8.6E-3 #Vtd
        if   pid0 in ["6","-6","3","-3"    ] and pid1 in ["6","-6","3","-3"    ]: return 41.5E-3 #Vts
        if   pid0 in ["6","-6","5","-5"    ] and pid1 in ["6","-6","5","-5"    ]: return 1.014 #Vtb

        elif pid0 in ['411','-411'] and pid1 in ['311','-311']: return 0.975
        elif pid0 in ['421','-421'] and pid1 in ['321','-321']: return 0.975
        elif pid0 in ['521','-521'] and pid1 in ['421','-421']: return 40.8E-3
        elif pid0 in ['511','-511'] and pid1 in ['411','-411']: return 40.8E-3
        elif pid0 in ['531','-531'] and pid1 in ['431','-431']: return 40.8E-3
        elif pid0 in ['541','-541'] and pid1 in ['511','-511']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['531','-531']: return 0.975
        elif pid0 in ['421','-421'] and pid1 in ['323','-323']: return 0.975
        elif pid0 in ['521','-521'] and pid1 in ['423','-423']: return 40.8E-3
        elif pid0 in ['511','-511'] and pid1 in ['413','-413']: return 40.8E-3
        elif pid0 in ['531','-531'] and pid1 in ['433','-433']: return 40.8E-3
        elif pid0 in ['541','-541'] and pid1 in ['513','-513']: return 0.221
        elif pid0 in ['541','-541'] and pid1 in ['533','-533']: return 0.975
        elif pid0 in ['130','310' ,"-130","-310"] and pid1 in ['211','-211']: return 0.2243

        #new channels pseudo
        elif pid0 in ['321','-321'] and pid1 in ['111','-111']: return V23
        elif pid0 in ['431','-431'] and pid1 in ['221','-221']: return V43
        elif pid0 in ['431','-431'] and pid1 in ['331','-331']: return V43
        elif pid0 in ['521','-521'] and pid1 in ['111','-111',"221","-221","331","-331"]: return V25 #not sure if its accurate for 221 and 331
        elif pid0 in ['541','-541'] and pid1 in ['421','-421']: return V25
        elif pid0 in ['541','-541'] and pid1 in ['441','-441']: return V45
        elif pid0 in ['421','-421'] and pid1 in ['211','-211']: return V41
        elif pid0 in ['411','-411'] and pid1 in ['111','-111',"221","-221","331","-331"]: return V41 #not sure if its accurate for 221 and 331
        elif pid0 in ['431','-431'] and pid1 in ['311','-311']: return V41
        elif pid0 in ['511','-511'] and pid1 in ['211','-211']: return V25
        elif pid0 in ['531','-531'] and pid1 in ['321','-321']: return V25

        #new channels vector
        elif pid0 in ['521','-521'] and pid1 in ['113','-113',"223","-223"]: return V25
        elif pid0 in ['541','-541'] and pid1 in ['443','-443']: return V45
        elif pid0 in ['421','-421'] and pid1 in ['213','-213']: return V41
        elif pid0 in ['411','-411'] and pid1 in ['113','-113',"223","-223"]: return V41 

        elif pid0 in ['411','-411'] and pid1 in ['313','-313']: return V43

        elif pid0 in ['431','-431'] and pid1 in ['313','-313']: return V41
        elif pid0 in ['431','-431'] and pid1 in ['333','-333']: return V43
        elif pid0 in ['511','-511'] and pid1 in ['213','-213']: return V25
        elif pid0 in ['531','-531'] and pid1 in ['323','-323']: return V25
        elif pid0 in ['541','-541'] and pid1 in ['423','-423']: return V25



    #3-body differential branching fraction dBr/(dq^2dE) for decay of pseudoscalar to pseudoscalar meson
    #pid0 is parent meson pid1 is daughter meson pid2 is lepton pid3 is HNL
    def get_3body_dbr_pseudoscalar(self,pid0,pid1,pid2):

        # read constant
        mH, mHp, mLep = self.masses(pid0), self.masses(pid1), self.masses(pid2)
        VHHp, tauH = self.VHHp(pid0,pid1), self.tau(pid0)
        SecToGev=1./(6.582122*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.166378*10**(-5) #GeV^(-2)

        #accounts for quark content of decay; only relevant for neutral mesons with several quark configurations
        cp=1

        #form factor parameters
        #D+ -> K0
        if pid0 in ["411","-411"] and pid1 in ["311","-311"]:
            #pidS, pidV = "431", "433"
            #f00, MV, MS = .747, 2.01027, 2.318      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            pidV, pidS = "433", "431"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D0 -> K+
        if pid0 in ["421","-421"] and pid1 in ["321","-321"]:
            #f00, MV, MS = .747, 2.01027, 2.318      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            pidV, pidS = "433", "431"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Ds+ -> K0
        if pid0 in ["431","-431"] and pid1 in ["311","-311"]:
            #f00, MV, MS = .747, 2.01027, 2.318      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            pidV, pidS = "413", "411"
            f00, MV, MS = .747, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/1511.04877.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B0 -> D+
        if pid0 in ["511","-511"] and pid1 in ["411","-411"]:
            #f00, MV, MS = 0.66, 6.400, 6.2749       #f00 obtained from https://arxiv.org/pdf/1505.03925v2.pdf
            pidV, pidS = "543", "541"
            f00, MV, MS = 0.66, self.masses(pidV), self.masses(pidS)       #f00 obtained from https://arxiv.org/pdf/1505.03925v2.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B+ -> D0
        if pid0 in ["521","-521"] and pid1 in ["421", "-421"]:
            #f00, MV, MS = 0.66, 6.400, 6.2749       #f00 obtained from https://arxiv.org/pdf/1505.03925v2.pdf
            pidV, pidS = "543", "541"
            f00, MV, MS = 0.66, self.masses(pidV), self.masses(pidS)       #f00 obtained from https://arxiv.org/pdf/1505.03925v2.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B0s -> Ds-
        if pid0 in ["531","-531"] and pid1 in ["431","-431"]:
            #pidV = "533"
            #f00, MV, MS = -0.65, self.masses(pidV), self.masses(pid0)          #f00 obtained from https://arxiv.org/pdf/1106.3003.pdf
            pidV, pidS = "543", "541"
            f00, MV, MS = -0.65, self.masses(pidV), self.masses(pidS)          #f00 obtained from https://arxiv.org/pdf/1106.3003.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)" 
        #Bc+ -> B0
        if pid0 in ["541","-541"] and pid1 in ["511","-511"]:
            #f00, MV, MS = -0.58, 6.400, 6.2749      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            pidV, pidS = "413", "411"
            f00, MV, MS = -0.58, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Bc+ -> B0s
        if pid0 in ["541","-541"] and pid1 in ["531","-531"]:
            #f00, MV, MS = -0.61, 6.400, 6.2749      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            pidV, pidS = "433", "431"
            f00, MV, MS = -0.61, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #Bc+ -> D0
        if pid0 in ["541","-541"] and pid1 in ["421","-421"]:
            #f00, MV, MS = 0.69, 6.400, 6.2749      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)      #f00 obtained from https://arxiv.org/pdf/hep-ph/0007169.pdf
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #K0L,K0s -> pi+
        if pid0 in ["130","310"] and pid1 in ["211","-211"]:
            cp=(1/2)
            f00, MV, MS = .9636, .878, 1.252        #f00 obtained from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.96.034501; pole masses are found on lbl for K_L
            #pidV, pidS = ""
            #f00, MV, MS = .9636, self.masses(pidV), self.masses(pidS)        #f00 obtained from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.96.034501; pole masses are found on lbl for K_L
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        
        #new modes
        #K+ to pi0; assuming same form factors as pi+
        if pid0 in ["321","-321"] and pid1 in ["111","-111"]:
            cp=(1/2)
            #pidV = "323"
            #f00, MV, MS = 0.970, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "313", "311"
            f00, MV, MS = 0.970, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"

        #this is neutral kaon
        #K0 to pi+; replaced by KL and KS
        #if pid0 in ["311","-311"] and pid1 in ["211","-211"]:
        #    f00, lambdap, lambda0, mpi = 0.970, 0.0267, 0.0117, self.masses(pid1)
        #    fp = str(f00) + "*(1+" + str(lambdap) + "*q**2/" + str(mpi) + "**2)"
        #    f0 = str(f00) + "*(1+" + str(lambda0) + "*q**2/" + str(mpi) + "**2)"

        #D^+_s \to \eta; assuming same form factors as pi+
        if pid0 in ["431","-431"] and pid1 in ["221","-221"]:
            #pidV = "433"
            #f00, MV, MS = 0.495, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "433", "431"
            f00, MV, MS = 0.495, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^+_s \to \eta'; assuming same form factors as pi+
        if pid0 in ["431","-431"] and pid1 in ["331","-331"]:
            #pidV = "433"
            #f00, MV, MS = 0.557, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "433", "431"
            f00, MV, MS = 0.557, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"

        #B^+ \to \pi^0; assuming same form factors as pi+
        if pid0 in [ "521","-521"] and pid1 in ["111","-111"]:
            cp=1/2
            #pidV ="523"
            #f00, MV, MS = 0.29, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "513","511"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B_c^+ \to \eta_c
        if pid0 in ["541","-541"] and pid1 in ["441","-441"]:
            #pidV = "543"
            #f00, MV, MS = 0.76, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "543","541"
            f00, MV, MS = 0.76, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^0 \to \pi^+
        if pid0 in ["421","-421"] and pid1 in ["211","-211"]:
            #pidV = "423"
            #f00, MV, MS = 0.69, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^- \to \pi^0 
        if pid0 in ["411","-411"] and pid1 in ["111","-111"]: 
            cp=(1/2)
            #pidV = "423"
            #f00, MV, MS = 0.69, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D^- \to \eta, \eta', used pi+ form factors and corrected by cp
        if pid0 in ["411","-411"] and pid1 in ["221","-221","331","-331"]:
            theta=-11.5*np.pi/180
            #eta; corrects for the fact that eta and etap are rotations of eta1 and eta8
            if pid1 in ["221","-221"]:
                cp = ((np.cos(theta)/np.sqrt(6))-(np.sin(theta)/np.sqrt(3)))**2
            #etap
            if pid1 in ["331","-331"]:
                cp = ((np.sin(theta)/np.sqrt(6)) + (np.cos(theta)/np.sqrt(3)))**2
            #pidV = "423"
            #f00, MV, MS = 0.69, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.69, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #D_s^+ \to K^0
        if pid0 in ["431","-431"] and pid1 in ["311","-311"]:
            #pidV = "433"
            #f00, MV, MS = 0.72, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "413", "411"
            f00, MV, MS = 0.72, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B^0 \to \pi^+
        if pid0 in ["511","-511"] and pid1 in ["211","-211"]:
            #pidV = "513"
            #f00, MV, MS = 0.29, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B- -> eta, etap
        if pid0 in ["521","-521"] and pid1 in ["221","-221","331","-331"]:
            theta=-11.5*np.pi/180
            #eta; corrects for the fact that eta and etap are rotations of eta1 and eta8
            if pid1 in ["221","-221"]:
                cp = ((np.cos(theta)/np.sqrt(6))-(np.sin(theta)/np.sqrt(3)))**2
            #etap
            if pid1 in ["331","-331"]:
                cp = ((np.sin(theta)/np.sqrt(6)) + (np.cos(theta)/np.sqrt(3)))**2
            #pidV ="523"
            #f00, MV, MS = 0.29, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.29, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        #B_s^0 \to K^+
        if pid0 in ["531","-531"] and pid1 in ["321","-321"]:
            #pidV = "533"
            #f00, MV, MS = 0.31, self.masses(pidV), self.masses(pid0)
            pidV, pidS = "523", "521"
            f00, MV, MS = 0.31, self.masses(pidV), self.masses(pidS)
            fp=str(f00)+"/(1-q**2/"+str(MV)+"**2)"
            f0=str(f00)+"/(1-q**2/"+str(MS)+"**2)"
        # prefactor
        prefactor=(cp)*tauH*VHHp**2*GF**2/(64*np.pi**3*mH**2)
        prefactor*=self.vcoupling[str(abs(int(pid2)))]**2
        fm="("+f0+"-"+fp+")*("+str(mH)+"**2-"+str(mHp)+"**2)/q**2" 
        #putting all terms together
        term1="("+fm+")**2*(q**2*(mass**2+"+str(mLep)+"**2)-(mass**2-"+str(mLep)+"**2)**2)"
        term2=f"2*("+fp+")*("+fm+")*mass**2*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term3=f"(2*("+fp+")*("+fm+")*"+str(mLep)+"**2*(4*energy*"+str(mH)+"+ "+str(mLep)+"**2-mass**2-q**2))"
        term4=f"("+fp+")**2*(4*energy*"+str(mH)+"+"+str(mLep)+"**2-mass**2-q**2)*(2*"+str(mH)+"**2-2*"+str(mHp)+"**2-4*energy*"+str(mH)+"-"+str(mLep)+"**2+mass**2+q**2)"
        term5=f"-("+fp+")**2*(2*"+str(mH)+"**2+2*"+str(mHp)+"**2-q**2)*(q**2-mass**2-"+str(mLep)+"**2)"
        bra=str(prefactor)  + "* coupling**2 *(" + term1   + "+(" + term2  + "+" + term3 + ")+("  + term4   + "+" + term5 + "))"
        return(bra)

    #3-body differential branching fraction dBr/(dq^2dE) for decay of pseudoscalar to vector meson
    #pid0 is parent meson pid1 is daughter meson pid2 is lepton pid3 is HNL
    def get_3body_dbr_vector(self,pid0,pid1,pid2):
        tauH=self.tau(pid0)
        SecToGev=1./(6.58*pow(10.,-25.))
        tauH=tauH*SecToGev
        GF=1.1663787*10**(-5)
        VHV=self.VHHp(pid0,pid1)
        #accounts for quark content of decay; only relevant for neutral mesons with several quark configurations
        cv=1
        #'D^0 -> K*^- + e^+ + N' form factors
        if pid0 in ['-421','421'] and pid1 in ['323','-323']:
            #Mp=1.969; MV=2.11
            pidp = "431"
            pidV ="433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=.76; s1A0=.17; s2A0=0; V0=1.03; s1V=.27; s2V=0; A10=.66; s1A1=.3     #from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006 (table IV)
            s2A1=.2*0; A20=.49; s1A2=.67; s2A2=.16*0        
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #'D^- -> K^{0*} + e^- + N' form factors
        if pid0 in ['-411','411'] and pid1 in ['313','-313']:
            #Mp=1.969; MV=2.11
            pidp = "431"
            pidV = "433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=.76; s1A0=.17; s2A0=0; V0=1.03; s1V=.27; s2V=0; A10=.66; s1A1=.3     #from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006 (table IV)
            s2A1=.2*0; A20=.49; s1A2=.67; s2A2=.16*0        
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #'B^+ -> \bar{D}*^0 + e^+ + N'
        if (pid0 in ['521','-521'] and pid1 in ['423','-423']):
            #Mp=6.277; MV=6.842
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.69; s1A0=0.58; s2A0=0; V0=0.76; s1V=0.57; s2V=0; A10=0.66; s1A1=0.78      #from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006 (table X)
            s2A1=0; A20=0.62; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^0 -> D*^- + e^+ + N' form factors
        if (pid0 in ['511','-511'] and pid1 in ['413','-413']):
            #Mp=6.277; MV=6.842
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.69; s1A0=0.58; s2A0=0; V0=0.76; s1V=0.57; s2V=0; A10=0.66; s1A1=0.78      #from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006 (table X)
            s2A1=0; A20=0.62; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^0_s -> D^*_s^- + e^+ + N' form factors
        if pid0 in ['531','-531'] and pid1 in ['433','-433']:
            #Mp=6.272; MV=6.332
            pidp = "541"
            pidV = "543"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.67; s1A0=0.35; s2A0=0; V0=0.95; s1V=0.372         #from https://arxiv.org/pdf/1212.3167.pdf (Table 1)
            s2V=0; A10=0.70; s1A1=0.463; s2A1=0; A20=0.75; s1A2=1.04; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #'B^+_c -> B*^0 + e^+ + N' form factors
        if pid0 in ['541','-541'] and pid1 in ['513','-513']:
            A00=-.27; mfitA0=1.86; deltaA0=.13; V0=3.27; mfitV=1.76; deltaV=-.052       #from https://arxiv.org/pdf/hep-ph/0007169.pdf (Table 3)
            A10=.6; mfitA1=3.44; deltaA1=-1.07; A20=10.8; mfitA2=1.73; deltaA2=-0.09
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #'B^+_c -> B^*_s^0+ e^+ + N' form factors
        if pid0 in ['541','-541'] and pid1 in ['533','-533']:
            A00=-.33; mfitA0=1.86; deltaA0=.13; V0=3.25; mfitV=1.76; deltaV=-.052       #from https://arxiv.org/pdf/hep-ph/0007169.pdf (Table 3)
            A10=.4; mfitA1=3.44; deltaA1=-1.07; A20=10.4; mfitA2=1.73; deltaA2=-0.09
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #new modes
        #B+ to rho^0 or omega
        if pid0 in ["521","-521"] and pid1 in ["113","-113","223","-223"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            cv=(1/2)
            #Mp=self.masses(pid0)
            #MV=self.masses('523')
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.73; s2A1=0.1; A20=0.29; s1A2=1.4; s2A2=0.5
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #Bc+ to J/psi
        if pid0 in ['541','-541'] and pid1 in ['443','-443']:
            A00=0.68; mfitA0=8.20; deltaA0=1.40; V0=0.96; mfitV=5.65; deltaV= 0.0013      #from https://arxiv.org/pdf/hep-ph/0007169.pdf (Table 3)
            A10=0.68; mfitA1=5.91; deltaA1=0.052; A20=-0.004; mfitA2=5.67; deltaA2=-0.004
            A0=f"({A00}/(1-(q**2/{mfitA0}**2)-{deltaA0}*(q**2/{mfitA0}**2)**2))"
            V=f"({V0}/(1-(q**2/{mfitV}**2)-{deltaV}*(q**2/{mfitV}**2)**2))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-(q**2/{mfitA1}**2)-{deltaA1}*(q**2/{mfitA1}**2)**2))"
            A2=f"({A20}/(1-(q**2/{mfitA2}**2)-{deltaA2}*(q**2/{mfitA2}**2)**2))"
        #\bar{D^0} to \rho^+
        if pid0 in ["421","-421"] and pid1 in ["213","-213"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            #Mp=self.masses(pid0); MV=self.masses('423')
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D^- to \rho^0
        if pid0 in ["411","-411"] and pid1 in ["113","-113"]:      #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            cv=(1/2)
            #Mp=self.masses(pid0); MV=self.masses('413')
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D^- to \omega
        if pid0 in ["411","-411"] and pid1 in ["223","-223"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            cv=(1/2)
            #Mp=self.masses(pid0); MV=self.masses('423')
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.66; s1A0=0.36; s2A0=0; V0=0.90; s1V=0.46         
            s2V=0; A10=0.59; s1A1=0.50; s2A1=0; A20=0.49; s1A2=0.89; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"

        #D_s^- \to \bar{K^{*0}}
        if pid0 in ["431","-431"] and pid1 in ["313","-313"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            #Mp=self.masses(pid0); MV=self.masses('433')
            pidp = "411"
            pidV = "413"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.67; s1A0=0.2; s2A0=0; V0=1.04; s1V=0.24        
            s2V=0; A10=0.57; s1A1=0.29; s2A1=0.42; A20=0.42; s1A2=0.58; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #D_s^- \to \phi
        if pid0 in ["431","-431"] and pid1 in ["333","-333"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            #Mp=self.masses(pid0); MV=self.masses('433')
            pidp = "431"
            pidV = "433"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.73; s1A0=0.10; s2A0=0; V0=1.10; s1V=0.26         
            s2V=0; A10=0.64; s1A1=0.29; s2A1=0; A20=0.47; s1A2=0.63; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B^0 \to \rho^-
        if pid0 in ["511","-511"] and pid1 in ["213","-213"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            #Mp=self.masses(pid0); MV=self.masses('513')
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.54; s2A1=0.1; A20=0.24; s1A2=1.40; s2A2=0.50
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B^- \to \omega; used rho- form factors
        if pid0 in ["511","-511"] and pid1 in ["223","-223"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            cv=(1/2)
            #Mp=self.masses(pid0); MV=self.masses('513')
            pidp = "511"
            pidV = "513"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.30; s1A0=0.54; s2A0=0; V0=0.31; s1V=0.59         
            s2V=0; A10=0.26; s1A1=0.54; s2A1=0.1; A20=0.24; s1A2=1.40; s2A2=0.50
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B_s^0 \to K^*-
        if pid0 in ["531","-531"] and pid1 in ["323","-323"]:       #https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.014006
            #Mp=self.masses(pid0); MV=self.masses('533')
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.37; s1A0=0.60; s2A0=0.16; V0=0.38; s1V=0.66         
            s2V=0.30; A10=0.29; s1A1=0.86; s2A1=0.6; A20=0.26; s1A2=1.32; s2A2=0.54
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))"
        #B_c^+ \to D^{*0} 
        if pid0 in ["541","-541"] and pid1 in ["423","-423"]:
            #Mp=self.masses(pid0); MV=self.masses('543')
            pidp = "521"
            pidV = "523"
            Mp = self.masses(pidp); MV = self.masses(pidV)
            A00=0.56; s1A0=0; s2A0=0; V0=0.98; s1V=0        #not sure if this is correct
            s2V=0; A10=0.64; s1A1=1; s2A1=0; A20=-1.17; s1A2=1; s2A2=0
            A0=f"({A00}/((1-q**2/{Mp}**2)*(1-({s1A0}*q**2/{Mp}**2)+({s2A0}*q**4/{Mp}**4))))"
            V=f"({V0}/((1-q**2/{MV}**2)*(1-({s1V}*q**2/{MV}**2)+({s2V}*q**4/{MV}**4))))"
            #form factors for A1 and A2
            A1=f"({A10}/(1-({s1A1}*q**2/{MV}**2)+({s2A1}*q**4/{MV}**4)))"
            A2=f"({A20}/(1-({s1A2}*q**2/{MV}**2)+({s2A2}*q**4/{MV}**4)))" 


        #form factors
        f1=f"({V}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f2=f"((self.masses('{pid0}')+self.masses('{pid1}'))*{A1})"
        f3=f"(-{A2}/(self.masses('{pid0}')+self.masses('{pid1}')))"
        f4=f"((self.masses('{pid1}')*(2*{A0}-{A1}-{A2})+self.masses('{pid0}')*({A2}-{A1}))/q**2)"
        f5=f"({f3}+{f4})"
        #s1A0 is sigma_1(A0) etc.
        omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2+m3**2-self.masses('{pid2}')**2-2*self.masses('{pid0}')*energy)"
        Omegasqr=f"(self.masses('{pid0}')**2-self.masses('{pid1}')**2-q**2)"
        prefactor=f"(({cv})*({tauH}*coupling**2*{VHV}**2*{GF}**2)/(32*np.pi**3*self.masses('{pid0}')**2))*{self.vcoupling[str(abs(int(pid2)))]}**2"
        term1=f"({f2}**2/2)*(q**2-m3**2-self.masses('{pid2}')**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term2=f"({f5}**2/2)*(m3**2+self.masses('{pid2}')**2)*(q**2-m3**2+self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term3=f"2*{f3}**2*self.masses('{pid1}')**2*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)*(m3**2+self.masses('{pid2}')**2-q**2+{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2))"
        term4=f"2*{f3}*{f5}*(m3**2*{omegasqr}+({Omegasqr}-{omegasqr})*self.masses('{pid2}')**2)*(({Omegasqr}**2/(4*self.masses('{pid1}')**2))-q**2)"
        term5=f"2*{f1}*{f2}*(q**2*(2*{omegasqr}-{Omegasqr})+{Omegasqr}*(m3**2-self.masses('{pid2}')**2))"
        term6=f"({f2}*{f5}/2)*({omegasqr}*({Omegasqr}/self.masses('{pid1}')**2)*(m3**2-self.masses('{pid2}')**2)+({Omegasqr}**2/self.masses('{pid1}')**2)*self.masses('{pid2}')**2+2*(m3**2-self.masses('{pid2}')**2)**2-2*q**2*(m3**2+self.masses('{pid2}')**2))"
        term7=f"{f2}*{f3}*({Omegasqr}*{omegasqr}*(({Omegasqr}-{omegasqr})/self.masses('{pid1}')**2)+2*{omegasqr}*(self.masses('{pid2}')**2-m3**2)+{Omegasqr}*(m3**2-self.masses('{pid2}')**2-q**2))"
        term8=f"{f1}**2*({Omegasqr}**2*(q**2-m3**2+self.masses('{pid2}')**2)-2*self.masses('{pid1}')**2*(q**4-(m3**2-self.masses('{pid2}')**2)**2)+2*{omegasqr}*{Omegasqr}*(m3**2-q**2-self.masses('{pid2}')**2)+2*{omegasqr}**2*q**2)"
        bra=str(prefactor) + "*(" + term1 + "+" + term2 + "+" + term3 + "+" + term4 + "+" + term5 + "+" + term6 + "+" + term7 + "+" + term8 + ")"
        return(bra)

    #pid0 is tau, pid1 is produced lepton and pid2 is the neutrino
    #3-body differential branching fraction dBr/(dE) for 3-body leptonic decay of tau lepton
    def get_3body_dbr_tau(self,pid0,pid1,pid2):
        if pid2=='16' or pid2=='-16':
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*{GF}**2*coupling**2*self.masses('{pid0}')**2*energy/(2*np.pi**3))*{self.vcoupling[str(abs(int(pid1)))]}**2"
            dbr=f"{prefactor}*(1+((mass**2-self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-2*(energy/self.masses('{pid0}')))*(1-(self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}'))))*np.sqrt(energy**2-mass**2)"
        else:
            SecToGev=1./(6.582122*pow(10.,-25.))
            tautau=self.tau(pid0)*SecToGev
            GF=1.166378*10**(-5) #GeV^(-2)
            prefactor=f"({tautau}*{GF}**2*coupling**2*self.masses('{pid0}')**2/(4*np.pi**3))*{self.vcoupling[str(abs(int(pid0)))]}**2"
            dbr=f"{prefactor}*(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}')))**2*np.sqrt(energy**2-mass**2)*((self.masses('{pid0}')-energy)*(1-(mass**2+self.masses('{pid1}')**2)/self.masses('{pid0}')**2)-(1-self.masses('{pid1}')**2/(self.masses('{pid0}')**2+mass**2-2*energy*self.masses('{pid0}')))*((self.masses('{pid0}')-energy)**2/self.masses('{pid0}')+((energy**2-mass**2)/(3*self.masses('{pid0}')))))"
        return(dbr)
    
    #############HNL decay channel branching ratios############
    #NOTE: the branching ratios for HNL given below are actually decay widths
    #corresponds lepton pids to a greek index ranging from 1 to 3
    def analyze_pid1_pid3(self,pid1,pid3):
        if pid1 in ["11","-11"]:
            beta=1
        if pid1 in ["13","-13"]:
            beta=2
        if pid1 in ["15","-15"]:
            beta=3
        if pid3 in ["12","-12"]:
            alpha=1
        if pid3 in ["14","-14"]:
            alpha=2
        if pid3 in ["16","-16"]:
            alpha=3
        return(alpha,beta)

    #for N->l_beta^+ l_beta^- nu_alpha decay width
    def calc_br_lb_lb_nua(self,pid1,pid2,pid3):
        GF=1.166378*10**(-5)
        delta=lambda l1,l2: 1 if l1==l2 else 0
        xl=f"(self.masses(pid1)/mass)"
        xw=f"(0.231)" #xw=sin(theta_w)^2
        L=f"np.log((1-3*{xl}**2-(1-{xl}**2)*np.sqrt(1-4*{xl}**2))/({xl}**2*(1+np.sqrt(1-4*{xl}**2))))"
        C1=f"(1/4)*(1-4*{xw}+8*{xw}**2)"
        C2=f"(1/2)*{xw}*(2*{xw}-1)"
        C3=f"(1/4)*(1+4*{xw}+8*{xw}**2)"
        C4=f"(1/2)*{xw}*(2*{xw}+1)"
        alpha,beta=analyze_pid1_pid3(pid1,pid3)
        coupling=f"self.vcoupling[str(abs(int({pid3}))-1)]"
        br_lb_lb_nua=f"({GF}**2*mass**5/(192*np.pi**3))*{coupling}**2*(({C1}*(1-{delta(alpha,beta)})+{C3}*{delta(alpha,beta)})*((1-14*{xl}**2-2*{xl}**4-12*{xl}**6)*np.sqrt(1-4*{xl}**2)+12*{xl}**4*({xl}**4-1)*{L})+4*({C2}*(1-{delta(alpha,beta)})+{C4}*{delta(alpha,beta)})*({xl}**2*(2+10*{xl}**2-12*{xl}**4)*np.sqrt(1-4*{xl}**2)+6*{xl}**4*(1-2*{xl}**2+2*{xl}**4)*{L}))"
        return(br_lb_lb_nua)

    #for N->l_beta^+ l_beta^- nu_alpha decay width
    lamda=lambda a,b,c: a**2+b**2+c**2-2*a*b-2*b*c-2*c*a
    I1_integrand=lambda s,x,y,z: (12/s)*(s-x**2-y**2)*(1+z**2-s)*np.sqrt(lamda(s,x**2,y**2))*np.sqrt(lamda(1,s,z**2))
    I2_integrand=lambda s,x,y,z: (24*y*z/s)*(1+x**2-s)*np.sqrt(lamda(s,y**2,z**2))*np.sqrt(lamda(1,s,x**2))
    from scipy.integrate import quad
    def calc_br_lb_lb_nua(self,pid1,pid2,pid3):
        xw=f"0.231"
        GF=1.166378*10**(-5)
        delta=lambda l1,l2: 1 if l1==l2 else 0
        if pid2=="11":
            l2=1
        if pid2=="13":
            l2=2
        if pid2=="15":
            l2=3
        if pid3=="12":
            l1=1
        if pid3=="14":
            l1=2
        if pid3=="16":
            l1=3
        gL=f"(-(1/2)+{xw})"
        gR=f"{xw}"
        x=f"0"
        y=f"(self.masses({pid2})/mass)"
        z=y
        I1=f"quad(I1_integrand,({x}+{y})**2,(1-{z})**2,args=({x},{y},{z}))[0]"
        I2=f"quad(I2_integrand,({y}+{z})**2,(1-{x})**2,args=({x},{y},{z}))[0]"
        coupling=f"self.vcoupling[str(abs(int({pid3}))-1)]"
        br_lb_lb_nua=f"({coupling}**2*{GF}**2*mass**5/(96*np.pi**3))*(({gL}*{gR}+{delta(l1,l2)}*{gR})*{I2}+({gL}**2+{gR}**2+{delta(l1,l2)}*(1+2*{gL}))*{I1})"
        return(br_lb_lb_nua)
    

    #for N->U bar{D} l_alpha^- decay width (inclusive mode)
    I_integrand=lambda s,x,y,z: (12/s)*(s-x**2-y**2)*(1+z**2-s)*np.sqrt(lamda(s,x**2,y**2))*np.sqrt(lamda(1,s,z**2))
    lamda=lambda a,b,c: a**2+b**2+c**2-2*a*b-2*b*c-2*c*a
    def calc_br_u_bd_l(self,pid1,pid2,pid3):
        GF=1.166378*10**(-5)
        xd=f"(self.masses({pid2})/mass)"
        xu=f"(self.masses({pid1})/mass)"
        xl=f"(self.masses({pid3})/mass)"
        x=xl
        y=xu
        z=xd
        I=f"quad(I_integrand,({x}+{y})**2,(1-{z})**2,args=({x},{y},{z}))[0]"
        coupling=f"self.vcoupling[str(abs(int({pid3})))]"
        br_u_d_l=f"(self.VHHp({pid1},{pid2})**2*{GF}**2*mass**5*coupling**2/(32*np.pi**3))*{I}"
        return(br_u_d_l)
    
    lamda=lambda a,b,c: a**2+b**2+c**2-2*a*b-2*b*c-2*c*a
    x=0
    xq=f"(self.masses(pid1)/mass)"
    y=xq
    z=xq
    xw=f"(0.231)" #xw=sin(theta_w)^2
    I1_integrand=lambda s,x,y,z: (12/s)*(s-x**2-y**2)*(1+z**2-s)*np.sqrt(lamda(s,x**2,y**2))*np.sqrt(lamda(1,s,z**2))
    I2_integrand=lambda s,x,y,z: (24*y*z/s)*(1+x**2-s)*np.sqrt(lamda(s,y**2,z**2))*np.sqrt(lamda(1,s,x**2))
    def calc_br_q_bq_nu(self,pid1,pid3):
        GF=1.166378*10**(-5)
        if pid1 in quarks_u:
            gL=f"(1/2-(2/3)*{xw})"
            gR=f"(-(2/3)*{xw})" #article had 2 g_R^U so I assumed one was supposed to have a D
        if pid1 in quarks_d:
            gL=f"(-1/2+(1/3)*{xw})"
            gR=f"((1/3)*{xw})"
        I1=f"quad(I1_integrand,({x}+{y})**2,(1-{z})**2,args=({x},{y},{z}))[0]"
        I2=f"quad(I2_integrand,({y}+{z})**2,(1-{x})**2,args=({x},{y},{z}))[0]"
        coupling=f"self.vcoupling[str(abs(int(pid3))-1)]"
        br_q_bq_nu=f"(coupling**2*{GF}**2*mass**5/(32*np.pi**3))*({gL}*{gR}*{I2}+({gL}**2+{gR}**2)*{I1})"
        return(br_q_bq_nu)
    ################################################################################################

    #input path to given folder and it removes all files in that folder
    def remove_files_from_folder(self,path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)


    def get_br_and_ctau(self,mpts):
        """
        
        Generate Decay Data and save to Decay Data directory
        
        """
        
        #define coupling tuple
        coupling = (self.vcoupling['11'],self.vcoupling['13'],self.vcoupling['15'])
        
        #create HNL_Decay Object
        Decay = HNL_Decay(couplings = coupling)
        
        #Generate ctau (stored in Decay.ctau)
        Decay.gen_ctau(mpts)
        
        #Generate branching ratios (stored in Decay.model_brs)
        Decay.gen_brs()
        
        #Write ctau, branching ratios, and decay widths to the Decay Width directory
        Decay.save_data(True,True,True)
        
        
    def set_brs(self):
        """
        
        Create list of decay modes, final states, and the location of thhe respective branching fraction to be passed to FORESEE
        """
        #define coupling tuple
        coupling = (self.vcoupling['11'],self.vcoupling['13'],self.vcoupling['15'])
        
        #create HNL_Decay Object
        Decay = HNL_Decay(couplings = coupling)
        
        modes =  [] 
        filenames = []
        #iterate over decay modes
        for channel in Decay.modes_active.keys():
            
            for mode in Decay.modes_active[channel]: 
                
                modes.append(mode)
                
                csv_path = fr"Decay Data/{(Decay.U['e'],Decay.U['mu'],Decay.U['tau'])}/br/{channel}/{mode}.csv"
                
                filenames.append(csv_path)
        
        finalstates  = []
        
        for mode in modes: 
            finalstate = [pid(i) for i in mode]
            
            finalstates.append(finalstate)
            
        
        return modes,finalstates,filenames
            
        
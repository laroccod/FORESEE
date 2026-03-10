#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Foresee.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility,Model,Foresee,LorentzVector,Decay
import pytest
import numpy as np
import random

foresee = Foresee(path=src_path)
foresee.rng.seed(137)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_threebody_decay_pure_phase_space():
    ref = [{'px':-4.524630, 'py': 37.565497, 'pz': 12.005886, 'e':60.508054},\
           {'px':11.575480, 'py':  5.151148, 'pz':  6.659295, 'e':26.948718},\
           {'px':-7.050850, 'py':-42.716645, 'pz':-18.665182, 'e':49.543227},]
    m0 = 137.
    p0=LorentzVector(px=0., py=0., pz=0., e=m0)
    ret = foresee.threebody_decay_pure_phase_space(p0=p0, m0=m0, m1=m0/3., m2=m0/6., m3=m0/9.)
    assert len(ret)==len(ref)
    for a,b in zip(ref,ret):
        assert np.isclose(a['px'], b.px, rtol=0.001)
        assert np.isclose(a['py'], b.py, rtol=0.001)
        assert np.isclose(a['pz'], b.pz, rtol=0.001)
        assert np.isclose(a['e' ], b.e,  rtol=0.001)
        
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_rng_init_requirement():
    
    #Check that the Utility random number generator cannot be accessed unless given in init
    decay_no_rng = Decay()
    rng_inaccessible = False
    try: decay_no_rng.rng.seed(137)
    except: rng_inaccessible = True
    assert rng_inaccessible
    
    #Now give an RNG in init, check that it can be accessed
    decay_rng = Decay(rng=random.Random())
    try:
        decay_rng.rng.seed(137)
        rng_accessible = True
    except: rng_accessible = False
    assert rng_accessible

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_rng_consistency():
    
    #Check that Utility/Decay random number generators are consistent w/ Foresee
    
    #Init a decay object with its own RNG, w/ same seed as Foresee
    #Simply checks consistency of Foresee RNG with a fresh random.Random() 
    foresee42 = Foresee(path=src_path)
    foresee42.rng.seed(42)
    decay42 = Decay(rng=random.Random())
    decay42.rng.seed(42)
    #Generate random sequences with both, assert same sequences were generated
    N_generate = 5
    rand_foresee42 = [foresee42.rng.uniform(0,1) for _ in range(N_generate)]
    rand_decay42   = [  decay42.rng.uniform(0,1) for _ in range(N_generate)]
    assert rand_foresee42==rand_decay42

    #A standalone decay initiated w/ same seed as Foresee must generate the same first event
    foresee137 = Foresee(path=src_path)
    foresee137.rng.seed(137)
    decay137 = Decay(rng=random.Random())
    decay137.rng.seed(137)
    m0 = 137.
    p0=LorentzVector(px=0., py=0., pz=0., e=m0)
    ret_f = foresee137.threebody_decay_pure_phase_space(p0=p0,m0=m0,m1=m0/3.,m2=m0/6.,m3=m0/9.)
    ret_d =   decay137.threebody_decay_pure_phase_space(p0=p0,m0=m0,m1=m0/3.,m2=m0/6.,m3=m0/9.)
    for a,b in zip(ret_f,ret_d):
        assert np.isclose(a.px, b.px, rtol=0.001)
        assert np.isclose(a.py, b.py, rtol=0.001)
        assert np.isclose(a.pz, b.pz, rtol=0.001)
        assert np.isclose(a.e,  b.e,  rtol=0.001)
    
    #Now ensure the same RNG instance is used in Foresee's Decay functions & other functions:
    #Advance rngs in both Foresee/Decay objects in different ways, but same number of steps
    tmp = decay137.decay_in_restframe_2body(br=0.5, m0=m0, m1=m0/2., m2=m0/4., nsample=1)
    tmp = [foresee137.rng.uniform(0,1) for _ in range(2)]  #Above calls rng twice
    #The threebody decay events generated next should again agree
    ret_f2 = foresee137.threebody_decay_pure_phase_space(p0=p0,m0=m0,m1=m0/3.,m2=m0/6.,m3=m0/9.)
    ret_d2 = decay137.threebody_decay_pure_phase_space(p0=p0,m0=m0,m1=m0/3.,m2=m0/6.,m3=m0/9.)
    for a,b in zip(ret_f2,ret_d2):
        assert np.isclose(a.px, b.px, rtol=0.001)
        assert np.isclose(a.py, b.py, rtol=0.001)
        assert np.isclose(a.pz, b.pz, rtol=0.001)
        assert np.isclose(a.e,  b.e,  rtol=0.001)
    

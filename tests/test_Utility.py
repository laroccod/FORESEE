#Contains tests for the Utility class

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Utility.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility,Foresee
import random
import pytest
import numpy as np

util = Utility()

#List common pdg ids expected to appear in FORESEE computations
testpdgids = [2112, -2112, 2212, -2212, 211  ,-211 , 321 , -321 ,\
              310 ,  130 , 111        , 221        , 331        ,\
              3122, -3122, 3222, -3222, 3112 ,-3112, 3322, -3322,\
              3312, -3312, 3334, -3334, 113        , 223        ,\
              333        , 213 , -213 , 411  ,-411 , 421 , -421 ,\
              431  ,-431 , 4122, -4122, 511  ,-511 , 521 , -521 ,\
              531  ,-531 , 541 , -541 , 5122 ,-5122, 4   , -4   ,\
              5    ,-5   , 11  , -11  , 13   ,-13  , 15  , -15  ,\
              22         , 23         , 24   ,-24  , 25         ,\
              0          , 443        , 100443      , 553       ,\
              100553     , 200553     , 12, -12, 14, -14, 16, -16]

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_charges():
    
    #Photons and neutrinos should have no charge
    for id in [22,12,14,16]: assert util.charges(id)==0        
    
    #Anti-particle charges must be 0 or opposite to particle charge
    for id in [negid for negid in testpdgids if negid<0]:        
        assert util.charges(id) in [0, -1.0*util.charges(abs(id))]  #Check both...
        assert util.charges(abs(id)) in [0, -1.0*util.charges(id)]  #...ways


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_masses():
    #Particle/anti-particle masses must agree
    for id in [negid for negid in testpdgids if negid<0]:        
        assert util.masses(id)==util.masses(abs(id))


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_utility_nans():    
    #Properties must not return not-a-number (nan)
    for id in testpdgids:
        assert not np.isnan(util.charges(id))
        assert not np.isnan(util.masses(id))
        assert not np.isnan(util.ctau(id))
        assert not np.isnan(util.widths(id))


#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_utility_infs():    
    #Properties must not return infinity (inf)
    for id in testpdgids:
        assert not np.isinf(util.charges(id))
        assert not np.isinf(util.masses(id))
        assert not np.isinf(util.ctau(id))
        assert not np.isinf(util.widths(id))

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_rng_init_requirement():
    
    #Check that the Utility random number generator cannot be accessed unless given in init
    util_no_rng = Utility()
    rng_inaccessible = False
    try: util_no_rng.rng.seed(137)
    except: rng_inaccessible = True
    assert rng_inaccessible
    
    #Now give an RNG in init, check that it can be accessed
    util_rng = Utility(rng=random.Random())
    try:
        util_rng.rng.seed(137)
        rng_accessible = True
    except: rng_accessible = False
    assert rng_accessible
        
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_rng_consistency():
    
    #Check that the Utility random number generator is consistent w/ Foresee
    #Init an Utility w/ its own RNG but same seed as Foresee, assert same sequences generated
    #Simply checks consistency of Foresee RNG with a fresh random.Random() 
    N_generate = 5  #Ask Foresee RNG to generate this many random numbers
    seed=42
    foresee_tmp = Foresee(path=src_path)
    foresee_tmp.rng.seed(seed)
    rand_foresee = [foresee_tmp.rng.uniform(0,1) for _ in range(N_generate)]
    util_tmp = Utility(rng=random.Random())
    util_tmp.rng.seed(seed)
    rand_util = [util_tmp.rng.uniform(0,1) for _ in range(N_generate)]
    assert rand_foresee==rand_util

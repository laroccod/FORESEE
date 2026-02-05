#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Foresee.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility,Model,Foresee,LorentzVector,Vector3D,LorentzArray,boostLorentzArray
import pytest
import numpy as np
#import import_ipynb

foresee = Foresee(path=src_path)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights_1():
    """
    Check the sum of xs values for pi+ using EPOSLHC & SIBYLL at various energies
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {'EPOSLHC_13.6_211': 1259075965736.8040,
           'SIBYLL_13.6_211': 968262425157.5983,
           'EPOSLHC_14_211': 1381338244262.9976,
           'SIBYLL_14_211': 980175192323.9988,
           'EPOSLHC_27_211': 1900779251078.6904,
           'SIBYLL_27_211': 1442273621493.8990,
           'EPOSLHC_100_211': 3494975701999.6006,
           'SIBYLL_100_211': 2390905532445.802
    }

    for energy in ["13.6","14","27","100"]:
        for generator in ["EPOSLHC","SIBYLL"]:
            for pid in ["211"]:
            
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights_2():
    """
    Check the sum of xs values for various light hadrons using EPOSLHC &
    SIBYLL & QGSJET at 14 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'EPOSLHC_14_111': 1540016521237.8992,
        'EPOSLHC_14_-211': 1354422658626.5984,
        'EPOSLHC_14_221': 168625794754.7002,
        'EPOSLHC_14_321': 163567242000.0,
        'EPOSLHC_14_310': 156484453688.50006,
        'EPOSLHC_14_223': 174369536956.30035,
        'SIBYLL_14_111': 1135037805475.7979,
        'SIBYLL_14_-211': 957309158089.9979,
        'SIBYLL_14_221': 193017427160.60062,
        'SIBYLL_14_321': 126329282125.99944,
        'SIBYLL_14_310': 122040615609.59941,
        'SIBYLL_14_223': 180030783726.6004,
        'QGSJET_14_111': 1620017887078.2053,
        'QGSJET_14_-211': 1337306648835.5032,
        'QGSJET_14_221': 240708917737.39948,
        'QGSJET_14_321': 158783905140.3007,
        'QGSJET_14_310': 155261268435.60068,
        'QGSJET_14_223': 0.0,
    }
    
    for energy in ["14"]:
        for generator in ["EPOSLHC"]:#,"SIBYLL","QGSJET"]:
            for pid in ["111","-211","221","321","310","223"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights_3():
    """
    Check the sum of xs values for various charm hadrons using NLO-P8 &
    DPMJET & Pythia8 & Pythia8-Forward at 13.6 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'NLO-P8_13.6_411': 1465801300.7249959,
        'NLO-P8_13.6_-421': 2833497451.5750012,
        'NLO-P8_13.6_4122': 211773228.74999905,
        'DPMJET_13.6_411': 757617140.1999979,
        'DPMJET_13.6_-421': 2402337888.900002,
        'DPMJET_13.6_4122': 221847601.10000074,
        'SIBYLL_13.6_411': 1526990065.6999984,
        'SIBYLL_13.6_-421': 3330583308.1999803,
        'SIBYLL_13.6_4122': 610248228.1000023,
        'Pythia8_13.6_411': 1763738803.700007,
        'Pythia8_13.6_-421': 3428131619.5999875,
        'Pythia8_13.6_4122': 249262647.19999886,
        'Pythia8-Forward_13.6_411': 1393189207.0,
        'Pythia8-Forward_13.6_-421': 2696044660.0,
        'Pythia8-Forward_13.6_4122': 1477589385.0
    }
    
    for energy in ["13.6"]:
        for generator in ["NLO-P8","DPMJET","SIBYLL","Pythia8","Pythia8-Forward"]:
            for pid in ["411","-421", "4122"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])
                    
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_read_list_momenta_weights_4():
    """
    Check the sum of xs values for various beauty hadrons using NLO-P8 &
    NLO-P8-min & NLO-P8-max at 13.6 TeV
    """
    
    #Expected values for the sum of flattened list_xs
    ref = {
        'NLO-P8_13.6_-511': 100674964.26881358,
        'NLO-P8_13.6_521': 101239418.50559343,
        'NLO-P8-Min_13.6_-511': 74871136.7904587,
        'NLO-P8-Min_13.6_521': 75275592.57858993,
        'NLO-P8-Max_13.6_-511': 150104244.28716186,
        'NLO-P8-Max_13.6_521': 150995882.83722088
    }
    
    for energy in ["13.6"]:
        for generator in ["NLO-P8","NLO-P8-Min","NLO-P8-Max"]:
            for pid in ["-511", "521"]:
                
                #Open file, fetch list of xs
                dirname = foresee.dirpath+"files/hadrons/"+energy+"TeV/"+generator+"/"
                filename = dirname+generator+"_"+energy+"TeV_"+pid+".txt"
                _, _,list_xs  = foresee.read_list_momenta_weights(filenames=[filename])
                
                #Check approx agreement of sum of flattened list_xs w/ expected ref
                assert np.isclose(sum(np.array(list_xs).flatten()),\
                                  ref[generator+"_"+energy+"_"+pid])

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_convert_list_to_momenta():
    """
    Check the correct formatting of the output momentum and weight lists/arrays
    """
    fname= "../files/hadrons/14TeV/NLO-P8/NLO-P8_14TeV_421.txt"
    foresee.rng.seed(137)
    p,wgt = foresee.convert_list_to_momenta(filenames=fname,\
                                            mass=foresee.masses("421"),\
                                            nsample=10)
    #Array types
    #assert type(p)==list  #rm if switching to new scikit vector.array eventually
    assert type(wgt)==np.ndarray
    
    #Dimensions
    assert len(p)==50480
    assert len(wgt)==len(p)
    
    #Element types
    assert type(p[0].px) in [np.float64,float]  #p must contain Lorentz vectors w/ float component px
    for w in wgt:
        assert type(w)==np.ndarray
        assert len(w)==1
        for wi in w: assert type(wi) in [np.float64,float]
    
    #Check first few weights
    assert np.isclose(wgt[ 0][0],18.1725)
    assert np.isclose(wgt[10][0],36.345 )
    assert np.isclose(wgt[20][0],36.345 )
    assert np.isclose(wgt[30][0],18.1725)

    #Check first few momenta
    p_ref = [{'px':-0.001859, 'py':-0.000909, 'pz':194.832, 'E':194.841},\
             {'px': 0.001684, 'py':-0.001067, 'pz':196.330, 'E':196.339},\
             {'px':-0.002082, 'py': 0.000127, 'pz':198.643, 'E':198.652},\
             {'px':-0.000247, 'py':-0.002055, 'pz':195.014, 'E':195.023},\
             {'px': 0.001073, 'py':-0.001501, 'pz':184.480, 'E':184.489}]
    for i in range(len(p_ref)):
        assert np.isclose(p[i].px, p_ref[i]['px'], rtol=0.01)
        assert np.isclose(p[i].py, p_ref[i]['py'], rtol=0.01)
        assert np.isclose(p[i].pz, p_ref[i]['pz'], rtol=0.01)
        assert np.isclose(p[i].e , p_ref[i]['E' ], rtol=0.01)

#TODO move vector-related tests to a new file, add tests for rotations, multiply, cross, dot, twobody decay

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_boost():
    
    #Test simple single vector boost
    p4 = LorentzVector(0.,0.,0.,0.5)
    bt = Vector3D(0.,0.,0.5)
    expect = LorentzVector(0.0, 0.0, -0.2886751, 0.5773502)
    
    p4_bt = p4.boost(bt)
    assert np.isclose(p4_bt.px, expect.px, rtol=0.01)
    assert np.isclose(p4_bt.py, expect.py, rtol=0.01)
    assert np.isclose(p4_bt.pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_bt.e,  expect.e,  rtol=0.01)
    
    #Test LorentzVector.boostvector: should have the same boostvector as bt
    bt4 = LorentzVector(0.,0.,1.,2.)
    p4_bt1 = p4.boost(bt4.boostvector)
    assert np.isclose(p4_bt1.px, expect.px, rtol=0.01)
    assert np.isclose(p4_bt1.py, expect.py, rtol=0.01)
    assert np.isclose(p4_bt1.pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_bt1.e,  expect.e,  rtol=0.01)

    #Now try a - sign in the spatial component
    bt_m = Vector3D(0.,0.,-0.5)
    bt4_m = LorentzVector(0.,0.,-1.,2.)
    expect_m = LorentzVector(0.0, 0.0, 0.2886751, 0.5773502)

    p4_bt_m = p4.boost(bt_m)
    assert np.isclose(p4_bt_m.px, expect_m.px, rtol=0.01)
    assert np.isclose(p4_bt_m.py, expect_m.py, rtol=0.01)
    assert np.isclose(p4_bt_m.pz, expect_m.pz, rtol=0.01)
    assert np.isclose(p4_bt_m.e,  expect_m.e,  rtol=0.01)
    
    p4_bt4_m = p4.boost(bt4_m.boostvector)
    assert np.isclose(p4_bt4_m.px, expect_m.px, rtol=0.01)
    assert np.isclose(p4_bt4_m.py, expect_m.py, rtol=0.01)
    assert np.isclose(p4_bt4_m.pz, expect_m.pz, rtol=0.01)
    assert np.isclose(p4_bt4_m.e,  expect_m.e,  rtol=0.01)
    
    #Check boostfactor vs boostvector order of operations
    assert (-1.*bt4_m).boostvector.z==bt4_m.boostvector.z
    assert -1.*bt4_m.boostvector.z==-bt4_m.boostvector.z

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_array_boost_single():
    
    #Repeat simple single vector boost test but using array boost methods
    expect = LorentzVector(0.0, 0.0, -0.2886751, 0.5773502)
    p4 = LorentzVector(0.,0.,0.,0.5)
    bt = Vector3D(0.,0.,0.5)
    p4bt = LorentzVector(0.,0.,0.5,1.)  #Boostvector will have z=1./2.=0.5, equal to bt
    p4_arr = LorentzArray({'px': [p4.px], 'py': [p4.py], 'pz': [p4.pz], 'energy': [p4.e]})
    b3_arr = LorentzArray({'x': [bt.x], 'y': [bt.y], 'z': [bt.z]})
    b4_arr = LorentzArray({'px': [p4bt.px], 'py': [p4bt.py], 'pz': [p4bt.z], 'energy': [p4bt.e]})
    boostfactor = 1.

    #Boost array of momenta by a single 3D boost
    p4_boosted_arr = boostLorentzArray(momenta=p4_arr,boostby=bt,boostfactor=boostfactor)    
    assert np.isclose(p4_boosted_arr[0].px, expect.px, rtol=0.01)
    assert np.isclose(p4_boosted_arr[0].py, expect.py, rtol=0.01)
    assert np.isclose(p4_boosted_arr[0].pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_boosted_arr[0].e,  expect.e,  rtol=0.01)
    
    #Boost array of momenta by a single 4D LorentzVector (extract boostvector, check application of boostfactor)
    p4_boosted_arr1 = boostLorentzArray(momenta=p4_arr,boostby=p4bt,boostfactor=boostfactor)    
    assert np.isclose(p4_boosted_arr1[0].px, expect.px, rtol=0.01)
    assert np.isclose(p4_boosted_arr1[0].py, expect.py, rtol=0.01)
    assert np.isclose(p4_boosted_arr1[0].pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_boosted_arr1[0].e,  expect.e,  rtol=0.01)

    #Boost array of momenta by array (3D boost element)
    p4_boosted_arr2 = boostLorentzArray(momenta=p4_arr,boostby=b3_arr,boostfactor=boostfactor)    
    assert np.isclose(p4_boosted_arr2[0].px, expect.px, rtol=0.01)
    assert np.isclose(p4_boosted_arr2[0].py, expect.py, rtol=0.01)
    assert np.isclose(p4_boosted_arr2[0].pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_boosted_arr2[0].e,  expect.e,  rtol=0.01)
    
    #Boost array of momenta by array (4D boost element)
    p4_boosted_arr3 = boostLorentzArray(momenta=p4_arr,boostby=b4_arr,boostfactor=boostfactor)    
    assert np.isclose(p4_boosted_arr3[0].px, expect.px, rtol=0.01)
    assert np.isclose(p4_boosted_arr3[0].py, expect.py, rtol=0.01)
    assert np.isclose(p4_boosted_arr3[0].pz, expect.pz, rtol=0.01)
    assert np.isclose(p4_boosted_arr3[0].e,  expect.e,  rtol=0.01)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_array_boost():
    
    #Test array boost methods for multiple particles and boosts
    arr_particle = [[0.,0.,0.,1.],\
                    [0.,0.,0.,2.],\
                    [0.,0.,0.,3.]]
    arr_boost3 = [[0.,0., 0.25],\
                  [0.,0., 0.50],\
                  [0.,0., 0.75]]
    arr_boost4 = [[0.,0., 1., 4.],\
                  [0.,0., 1., 2.],\
                  [0.,0., 3., 4.]]
    """
    N.B. for the above, the boostlist function produces the following
      All particles in 1st boost direction
      [ [-0.  0.25819889]
        [-0.  0.51639778]
        [-0.  0.77459667]
        All particles in 2nd boost direction
        [-0.  0.57735027]
        [-0.  1.15470054]
        [-0.  1.73205081]
        All particles in 3rd boost direction
        [-0.  1.13389342]
        [-0.  2.26778684]
        [-0.  3.40168026] ]
    TODO the current boostLorentzArray reflects the behavior of MomentumNumpy4D arrays,
    and returns three particles, each boosted into the direction of a dedicated vector.
    If this behavior is changed to correspond to boostlist instead, the test below may 
    be expanded to check all the numbers given above.
    """
    expect = [0.25819889, 1.15470054, 3.40168026]
    
    #Call array wrappers
    parT = np.array(arr_particle).T
    bst3T = np.array(arr_boost3).T 
    bst4T = np.array(arr_boost4).T 
    momenta  = LorentzArray({'px':  parT[0], 'py':  parT[1], 'pz':  parT[2], 'energy': parT[3]})
    boostby4 = LorentzArray({'px': bst4T[0], 'py': bst4T[1], 'pz': bst4T[2], 'energy':bst4T[3]})
    boostby3 = LorentzArray({ 'x': bst3T[0],  'y': bst3T[1],  'z': bst3T[2]})
    
    boostfactor = -1.  #N.B. typical use case is -1
    
    #Boost array of momenta by an array of 3-vector boosts
    boosted3 = boostLorentzArray(momenta=momenta,boostby=boostby3,boostfactor=boostfactor)
    assert np.isclose(boosted3[0].pz,expect[0],rtol=0.01)
    assert np.isclose(boosted3[1].pz,expect[1],rtol=0.01)
    assert np.isclose(boosted3[2].pz,expect[2],rtol=0.01)

    #Boost array of momenta by an array of 4-vectors (extract boostvectors and check application of boostfactor)
    boosted4 = boostLorentzArray(momenta=momenta,boostby=boostby4,boostfactor=boostfactor)
    assert np.isclose(boosted4[0].pz,expect[0],rtol=0.01)
    assert np.isclose(boosted4[1].pz,expect[1],rtol=0.01)
    assert np.isclose(boosted4[2].pz,expect[2],rtol=0.01)

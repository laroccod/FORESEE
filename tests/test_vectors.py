#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Foresee.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import LorentzVector,Vector3D,LorentzArray,boostLorentzArray
import pytest
import numpy as np

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_dot4():
    p4 = LorentzVector(0.,0.,1.,1)
    assert p4.dot(p4)==0.
    v = LorentzVector(1.,2.,3.,4)
    w = LorentzVector(1.,1.,1.,4)
    assert v.dot(w)==(16.-3.-2.-1.)
    
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_dot3():
    p3 = Vector3D(0.,0.,1.)
    assert p3.dot(p3)==1.
    v = Vector3D(1.,2.,3.)
    w = Vector3D(1.,1.,1.)
    assert v.dot(w)==(3.+2.+1.)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_cross():
    x = Vector3D(1.,0.,0.)
    y = Vector3D(0.,1.,0.)
    z = Vector3D(0.,0.,1.)
    assert x.cross(y)== z
    assert x.cross(z)==-y
    assert y.cross(z)== x
    xy1 = Vector3D(1.,4.,0.)
    xy2 = Vector3D(2.,3.,0.)
    assert xy1.cross(xy2)==Vector3D(0.,0.,3.-8.)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_angle():
    x = Vector3D(1.,0.,0.)
    y = Vector3D(0.,1.,0.)
    z = Vector3D(0.,0.,1.)
    null = Vector3D(0.,0.,0.)
    v=Vector3D(1.,1.,0.)
    assert v.angle(null)==0.
    assert np.isclose(v.angle(-v), np.pi,      rtol=0.0001)
    assert np.isclose(v.angle(x),  np.pi*0.25, rtol=0.0001)
    assert np.isclose(v.angle(y),  np.pi*0.25, rtol=0.0001)
    assert np.isclose(v.angle(z),  np.pi*0.5,  rtol=0.0001)
    w=Vector3D(-1.,1.,0.)
    assert np.isclose(w.angle(x),  np.pi*0.75, rtol=0.0001)  #Ensure smallest angle given, w/ + sign

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_mul():
    v = Vector3D(1.,2.,3.)
    assert -2.*v==Vector3D(-2.,-4.,-6)
    assert -2.*v==v*-2.
    p4 = LorentzVector(0.,0.,1.,1)
    p4m = -1.*p4
    p4m_r = p4*-1.
    assert p4m==LorentzVector(0.,0.,-1.,-1.)  #N.B. also t-component multiplied!
    assert p4m==p4m_r

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_add_sub():
    
    v = Vector3D(1.,2.,3.)
    vaddv = v+v
    vsubv = v-v
    assert type(vaddv)==type(v)
    assert type(vsubv)==type(v)
    assert vaddv==Vector3D(2.,4.,6.)
    assert vsubv==Vector3D(0.,0.,0.)

    p  = LorentzVector(px= 1.,py=0.,pz=0.,e=1.)
    pm = LorentzVector(px=-1.,py=0.,pz=0.,e=1.)
    paddp  = p+p
    psubp  = p-p
    paddpm = p+pm
    psubpm = p-pm
    assert type(paddp )==type(p)
    assert type(psubp )==type(p)
    assert type(paddpm)==type(p)
    assert type(psubpm)==type(p)
    assert paddp ==LorentzVector(px=2., py=0., pz=0., e=2.)  #Lightlike
    assert psubp ==LorentzVector(px=0., py=0., pz=0., e=0.)  #Nothing
    assert paddpm==LorentzVector(px=0., py=0., pz=0., e=2.)  #Timelike at rest
    assert psubpm==LorentzVector(px=2., py=0., pz=0., e=0.)  #Spacelike

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_rotate():
    x = Vector3D(1.,0.,0.)
    y = Vector3D(0.,1.,0.)
    z = Vector3D(0.,0.,1.)
    xp = x.rotate(0.5*np.pi,z)
    assert x.angle(Vector3D(1.,0.,0.))==0.  #Ensure original vector unchanged
    assert np.isclose(xp.angle(y),0.,0.0001)
    p4 = LorentzVector(3.,4.,0.,5)
    p4p = LorentzVector(0.,0.,p4.z,p4.e)
    p4p.rotate(p4.angle(z),y).rotate(p4.angle(y),z)
    assert np.isclose(p4p.angle(p4),0.,rtol=0.0001)
    
#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_unit_vector():
    v = LorentzVector(0.,3.,4.,8.)
    assert v.vector==Vector3D(0.,3.,4.)  #Magnitude sqrt(3^2+4^2)=5
    assert v.vector.unit()==Vector3D(0.,0.6,0.8)
    
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

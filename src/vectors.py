import numpy as np
import math
try:
    from skhep.math.vectors import LorentzVector, Vector3D
    _OLD_SKHEP = True
except:
    from vector import MomentumObject3D,MomentumObject4D
    from vector import VectorNumpy3D,VectorNumpy4D
    from vector import MomentumNumpy3D,MomentumNumpy4D
    from vector import array as skheparray
    _OLD_SKHEP = False    

##############################################
##############################################
#  Classes for skhep backwards compatibility
##############################################
##############################################

if not _OLD_SKHEP:

    class Vector3D(MomentumObject3D):
        def __init__(self, x, y, z):
            super().__init__(x=x,y=y,z=z)

        def angle(self, vec):
            """
            Angle between this 3-vector and another. Undefined if a vector is (0,0,0), default to 0
            """
            try:
                #If a skheparray of vectors given
                vec_len = len(vec)
                ret = self.deltaangle(vec)
                ret[np.isnan(ret)] = 0.  #Replace NaN values w/ zeroes
                return ret
            except:
                #A single vector given
                ret = self.deltaangle(MomentumObject3D(x=vec.x, y=vec.y, z=vec.z))
                return ret if not math.isnan(ret) else 0.

        #Override superclass function return value types

        def cross(self,vec):
            try:
                vec_len = len(vec)
                superobj = super().cross(skheparray({'x':vec.x,'y':vec.y,'z':vec.z}))
                return skheparray({'x':superobj.x,'y':superobj.y,'z':superobj.z})
            except:
                superobj = super().cross(vec)
                return Vector3D(x=superobj.x,y=superobj.y,z=superobj.z)

        def unit(self):
            superobj = super().unit()
            return Vector3D(x=superobj.x,y=superobj.y,z=superobj.z)

        #Override left and right multiplication s.t. we can do e.g. -1.*Vector3D
        def __mul__(self,val):
            ret = super().__mul__(val)
            return Vector3D(x=ret.x,y=ret.y,z=ret.z)
        def __rmul__(self,val):
            return self.__mul__(val)


        def tolist(self):
            """
            Turn a Vector3D object into a list of numbers
            """
            return [self.x, self.y, self.z]
        
    class LorentzVector(MomentumObject4D):
        
        def __init__(self, px, py, pz, e):
            super().__init__(x=px,y=py,z=pz,t=e)
                
        @property
        def vector(self):
            """
            Previous LorentzVector.vector returned a 3-momentum, not the whole 4-vector
            """
            return Vector3D(x=self.x, y=self.y, z=self.z)

        def rotate(self,angle,axis):
            superobj = super().rotate_axis(axis=axis,angle=angle)
            return LorentzVector(px=superobj.x,py=superobj.y,pz=superobj.z,e=superobj.t)
        
        #Override left and right multiplication s.t. we can do e.g. -1.*LorentzVector
        #Note however that this reproduces the behavior -1.*(x,y,z,t)=(-x,-y,-z,-t)
        #instead of flipping the spatial components, which is often useful for boosts
        def __mul__(self,val):
            ret = super().__mul__(val)
            return LorentzVector(px=ret.x,py=ret.y,pz=ret.z,e=ret.t)
        def __rmul__(self,val):
            return self.__mul__(val)
        
        @property
        def boostvector(self):
            """
            Previous LorentzVector.vector could also return a 3-vector useful for boosts
            
            N.B. vector package has many alternatives for boost(), and they cannot all be 
            overridden. Therefore, although boosting 4-vectors with other 4-vectors
            may be more optimal according to scikit vector documentation
            (https://vector.readthedocs.io/en/latest/src/vector4d.html)
            it is recommended to use LorentzVector.boost(Vector3D), which will internally
            turn 3D vector into 4D, call the relevant function in the super class, and ensure
            a LorentzVector is returned instead of an instance of the super class.
            """
            return Vector3D(x=self.x/self.t,y=self.y/self.t,z=self.z/self.t)

        def boost(self,vec3D):
            """
            Override superclass boost function return value type
            """
            superobj = super().boostCM_of_beta3(vec3D)
            return LorentzVector(px=superobj.x,py=superobj.y,pz=superobj.z,e=superobj.t)
        
        def tolist(self):
            """
            Turn a LorentzVector object into a list of numbers
            """
            return [self.px, self.py, self.pz, self.E]

        
##############################################
##############################################
#  AUX functions wrapping skhep version deps 
##############################################
##############################################

#TODO consider including these in a class instead of free functions

def boostvector_tolist(momentum):
    """
    Turn the boostvector of a LorentzVector object into a list of floats
    Parameters
    ----------
    momentum: LorentzVector
        The 4-vector whose boost vector to extract
    Returns
    -------
    A list of 3 floats
    """
    boostvec = momentum.boostvector
    return [boostvec.x, boostvec.y, boostvec.z]


def LorentzArray(compvecdict):
    """
    Construct particle 4-momentum list/array, or arrays of spatial 3D vectors
    Parameters
    ----------
    compvecdict: {str: list / np.array(float)}
        The components 
        E.g. {'px': [float,...], ..., 'energy': [float,...]}
             or
             {'pt': [float,...], 'theta': [float,...], 'phi': [float,...], 'energy': [float,...]}
             or
             {'x': [float,...], 'y': [float,...], 'phi': [float,...], 'energy': [float,...]}
    Returns
    -------
    A list of LorentzVectors (old skhep), or a skheparray (new skhep) technically corresponding to 
    MomentumNumpy4D or MomentumNumpy3D, depending on the input arguments
    """
    if _OLD_SKHEP:
        #4D
        if sum([key in compvecdict for key in ['px','py','pz','energy']])==4:
            return list(map(LorentzVector,compvecdict['px'    ],\
                                          compvecdict['py'    ],\
                                          compvecdict['pz'    ],\
                                          compvecdict['energy']))            
        elif sum([key in compvecdict for key in ['x','y','z','t']])==4:
            return list(map(LorentzVector,compvecdict['x'],\
                                          compvecdict['y'],\
                                          compvecdict['z'],\
                                          compvecdict['t']))            
        elif sum([key in compvecdict for key in ['pt','theta','phi','energy']])==4:
            return list(map(LorentzVector,np.multiply(compvecdict['pt'],np.cos(compvecdict['phi'  ])),\
                                          np.multiply(compvecdict['pt'],np.sin(compvecdict['phi'  ])),\
                                          np.divide(compvecdict['pt'],np.tan(compvecdict['theta'])),\
                                          compvecdict['energy']))
        #3D
        elif sum([key in compvecdict for key in ['x','y','z']])==3:
            return list(map(Vector3D,compvecdict['x'],compvecdict['y'],compvecdict['z']))            
        else:
            print('LorentzArray components must be px,py,pz,energy, returning empty list')
            return []
    else:
        return skheparray(compvecdict)
    
def LorentzVectors_to_f_arr(momenta,mode='4D',boostf=-1):
    """
    Turn a list of LorentzVector objects (old skhep) or skheparray 
    (new skhep) into np.array(float) suitable for efficient numerics

    Parameters
    ----------
    momenta: [[LorentzVector/Vector3D],...] / skheparray / [[float,float,float,float],...] / np.array
        A list of momentum vectors
    mode: string
        Indicates if the input momenta are '4D' or '3D' vectors, or 'boost' to return boostvectors 
    boostf: float
        Constant multiplier. Defaults to -1 often required for rest frame boosts
    Returns
    -------
    A numpy array of lists w/ 4 floats: np.array([ [px,py,pz,E], ... ])
    """
        
    if len(momenta)==0: return np.array([])
    if _OLD_SKHEP or type(momenta[0]) in [LorentzVector, Vector3D]:
        if mode=='boost':
            return boostf*np.array(list(map(lambda p: boostvector_tolist(p), momenta)))
        else:  #3D or 4D
            return np.array(list(map(lambda p: p.tolist(), momenta)))
    elif not _OLD_SKHEP:
        try:
            if mode=='boost':
                return boostf*np.array([momenta.to_beta3().x,momenta.to_beta3().y,momenta.to_beta3().z]).T
            elif mode=='3D':
                return np.array([momenta.x, momenta.y, momenta.z]).T
            else: #4D
                return np.array([momenta.px, momenta.py, momenta.pz, momenta.e]).T
        except: return np.array(momenta)
    else:
        return np.array(momenta)  #Already list or np.array

def theta_p3_f_arr(momenta):
    """
    Angles wrt z-axis and magnitudes of 3-momenta
    
    Parameters
    ----------
    momenta: [LorentzVector] / skheparray
        Array of momenta from which to extract the theta angles and 3-momentum magnitudes
    Returns
    -------
    A 2D np.array(float) w/ first index corresponding to particles: [ [theta, |p3|], ... ]
    """
    if _OLD_SKHEP:
        return np.array(list(map(lambda p: [np.arctan(p.pt/p.pz), p.p], momenta)))
    else:
        return np.array([momenta.theta, momenta.p]).T

def get_rotangle(refax,p4):
    """
    Fetch rotation angles for a 4-vector p4, or several momenta, based on a given reference axis

    Parameters
    ----------
    refax: Vector3D
        Reference axis, e.g. an object corresponding to the z-axis
    p4: LorentzVector / [LorentzVector] / skheparray
        The momentum / momenta from which to extract the angle
    Returns
    -------
        If p4 is a single momentum, the angle as a float, else an array of floats
    """
    if _OLD_SKHEP and type(p4)==list:
        return list(map(lambda p: refax.angle(p.vector), p4))
    elif type(p4)==LorentzVector: return refax.angle(p4.vector)
    #p0_ is a Vector3D or a skheparray, for which this works for array (3D or 4D) or a single p4
    else: return refax.angle(p4)

def get_rotaxis(refax,p4):
    """
    Fetch a rotation axis object for a 4-vector p4, or several momenta, based on a given reference axis

    Parameters
    ----------
    refax: Vector3D
        Reference axis, e.g. an object corresponding to the z-axis
    p4: LorentzVector / [LorentzVector] / skheparray
        The momentum / momenta from which to extract the axis
    Returns
    -------
        If p4 is a single momentum, the rotation axis as a Vector3D, 
        else [Vector3D] (old skhep) / skheparray (new skhep)
    """
    if _OLD_SKHEP and type(p4)==list:
        return list(map(lambda p: refax.cross(p.vector).unit(), p4))
    elif type(p4)==LorentzVector: return refax.cross(p4.vector).unit()
    #p0_ is a Vector3D or a skheparray, for which this works for array (3D or 4D) or a single p4
    else: return refax.cross(p4).unit()

def rotateLorentzArray(momenta,rotangle,rotaxis):
    """
    Rotate all vectors in an array of 4-momenta
    
    Parameters
    ----------
    momenta: [LorentzVector] / skheparray
        An array of vectors to be rotated
    rotangle: float
        The angle how much to rotate
    rotaxis: Vector3D
        The axis about which to rotate
    Returns
    -------
    The rotated vectors as a list of LorentzVectors (old skhep) / skheparray (new skhep)
    """
    if type(rotangle) in [float, np.float64] and rotangle==0: return momenta
    if _OLD_SKHEP:
        if type(rotangle) in [float, np.float64]:
            #Rotate every vector in momenta by the same angle about the same axis
            return list(map(lambda p: p.rotate(rotangle,rotaxis), momenta))
        elif type(rotaxis)==Vector3D:
            #Rotate each vector in momentum by dedicated angles about the same axis
            return list(map(lambda p, alpha: p.rotate(alpha,ax,rotaxis), momenta, rotangle))
        else:
            #Rotate each vector in momentum by dedicated angles and axes
            return list(map(lambda p, alpha, ax: p.rotate(alpha,ax), momenta, rotangle, rotaxis))
    else:
        #skheparray supports lists of angles / 3D vector skheparrays out-of-the-box (rotates elementwise)
        return momenta.rotate_axis(rotaxis,rotangle)

def boostLorentzArray(momenta,boostby,boostfactor):
    """
    Boost all 4-momenta in an array
    
    Parameters
    ----------
    momenta: [LorentzVector] / skheparray
        An array of vectors to be boosted
    boostby: Vector3D / [Vector3D] / LorentzVector / skheparray
        A vector by which to boost, see LorentzVector.boostvector.
    boostfactor: float
        Apply a factor to the boost vector. E.g. boosting to a particle's rest 
        frame typically requires a factor of -1.0
        N.B. to be applied on a 3-vector/boostvector, NOT a 4-vector as it would
        cancel when calling LorentzVector.boostvector, which yields the 3-vector
        components (x/t,y/t,z/t); multiplying a LorentzVector by the boostfactor
        also affects t in both old and new skhep vector implementations.
        See also the logic of boostlist and LorentzVectors_to_f_arr.
    Returns
    -------
    The boosted vectors as a list of LorentzVectors (old skhep)/skheparray (new)
    
    N.B. assumes an individual boostvector for each momentum instead of
         boosting all vectors into all boosts's directions like boostlist does!
    """
    if _OLD_SKHEP:
        #A single boostvector given
        if type(boostby)==Vector3D:
            return list(map(lambda p: p.boost(boostfactor*boostby), momenta))
        elif type(boostby)==LorentzVector:
            return list(map(lambda p: p.boost(boostfactor*(boostby.boostvector)), momenta))
        else:
            #Assume boostvector is a list / array
            try:
                return list(map(lambda p, beta: p.boost(boostfactor*beta), momenta, boostby))
            except:
                return list(map(lambda p, beta: p.boost(boostfactor*(beta.boostvector)), momenta, boostby))
    else:

        #Ensure wrapper class used instead of superobject
        if type(boostby)==MomentumObject4D:
            boostby = LorentzVector(px=boostby.x, py=boostby.y, pz=boostby.z, e=boostby.t)
        elif type(boostby)==MomentumObject3D:
            boostby = Vector3D(x=boostby.x, y=boostby.y, z=boostby.z)
        
        #Return values depending on input format
        if type(boostby)==LorentzVector:
            if type(momenta)==LorentzVector:
                return momenta.boost(boostfactor*(boostby.boostvector))
            else: return momenta.boostCM_of_beta3(boostfactor*(boostby.boostvector))
        elif type(boostby)==Vector3D:
            if type(momenta)==LorentzVector:
                return momenta.boost(boostfactor*boostby)
            else: return momenta.boostCM_of_beta3(boostfactor*boostby)
        elif type(boostby) in [VectorNumpy4D,MomentumNumpy4D]:
            return momenta.boostCM_of_beta3(skheparray({'x':boostfactor*boostby.x/boostby.t,\
                                                        'y':boostfactor*boostby.y/boostby.t,\
                                                        'z':boostfactor*boostby.z/boostby.t}))
        elif type(boostby) in [VectorNumpy3D,MomentumNumpy3D]:
            return momenta.boostCM_of_beta3(skheparray({'x':boostfactor*boostby.x,\
                                                        'y':boostfactor*boostby.y,\
                                                        'z':boostfactor*boostby.z}))
        else:
            print('WARNING boostLorentzArray: unspecified type '+str(type(boostby)))
            #Default solution: float array w/ first index corresponding to x,y,z,t
            return momenta.boostCM_of_beta3(skheparray({'x':boostfactor*boostby[0]/boostby[3],\
                                                        'y':boostfactor*boostby[1]/boostby[3],\
                                                        'z':boostfactor*boostby[2]/boostby[3]}))

## Import libraries

#import emg3d
import empymod
import numpy as np
from scipy.constants import mu_0
import time
import pygimli as pg

class EMforwardOpt_3lay(pg.Modelling):
    def __init__(self):
        """Initialize the model."""
        super().__init__()        
    def response(self, x):
        sig1 = x[0]
        sig2 = x[1]
        sig3 = x[2]
        thk1 = x[3]
        thk2 = x[4]
        if (thk1+thk2) >= 10:
            thk1 = 2
            thk2 = 3
        Z = EMforward3lay(sig1, sig2, sig3, thk1, thk2, height=0.47)                           
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [2,3]
        sig_ini = [1/20,1/20, 1/20]
        x0 = sig_ini + thk_ini
        return np.array(x0)

def EMforward2lay(sigma1, sigma2, thicks1, offsets = np.array([2, 4, 8]), height=0, freq=9000):
    """ This function performs the forward simulation of a the measurements in a
    low frequency electromagnetic induction device assuming a 2 layered earth
    using HCP, VCP and PRP geometries 
    
    Parameters:
    - sigma1: Conductivity of the first layer [S/m]
    - sigma2: Conductivity of the second layer [S/m]
    - thicks1: Thickness of the first layer [m]
    - offsets: separations of the transmitter and receiver coils [m]
    - height: height of the device with respect to ground [m]
    - freq: frequency of device [Hz]
    
    Returns:
    Array [Q_HCP, Q_VCP, Q_PRP, P_HCP, P_VCP, P_PRP]
    - Q_HCP : Quadrature Horizontal Coplanar
    - Q_VCP : Quadrature Vertical Coplanar
    - Q_PRP : Quadrature Perpendicular
    - P_HCP : In-Phase Horizontal Coplanar
    - P_VCP : In-Phase Vertical Coplanar
    - P_PRP : In-Phase Perpendicular
    
    """  
    time.sleep(0.01)
    
    # Source and receivers geometry
    sx = 0
    sy = 0
   
    Hsource    = [sx, sy, -height]
    Hreceivers = [offsets, offsets*0, -height]

    Vsource    = [sx, sy, -height]
    Vreceivers = [offsets, offsets*0, -height]

    Psource    = [sx, sy, -height]
    Preceivers = [offsets+0.1, offsets*0, -height]
    
    res =[1e6, 1/sigma1, 1/sigma2]
    depth=[0, thicks1]

    # Compute fields
    HCP_Hs = empymod.dipole(Hsource, Hreceivers, depth, res, freq, 
                            ab=66, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
    VCP_Hs = empymod.dipole(Vsource, Vreceivers, depth, res, freq,
                            ab =55, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hs = empymod.dipole(Psource, Preceivers, depth, res, freq, 
                            ab=46, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 

    HCP_Hp = empymod.dipole(Hsource, Hreceivers, depth=[], res=[1e6], freqtime=freq,
                          ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
    VCP_Hp = empymod.dipole(Vsource, Vreceivers, depth=[], res=[1e6], freqtime=freq,
                          ab=55, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hp = empymod.dipole(Psource, Preceivers, depth=[], res=[1e6], freqtime=freq, 
                            ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
#
    Q_HCP = (HCP_Hs/HCP_Hp).imag.amp() 
    Q_VCP = (VCP_Hs/VCP_Hp).imag.amp() 
    Q_PRP = (PRP_Hs/PRP_Hp).imag.amp() 
    
    P_HCP = (HCP_Hs/HCP_Hp).real.amp() 
    P_VCP = (VCP_Hs/VCP_Hp).real.amp() 
    P_PRP = (PRP_Hs/PRP_Hp).real.amp() 
    
    return np.hstack((Q_HCP, Q_VCP, Q_PRP, P_HCP, P_VCP, P_PRP))

def EMforward2lay_field(sigma1, sigma2, thicks1, offsets = np.array([2, 4, 8]), height=0, freq=9000):
    time.sleep(0.01)
    
    # Source and receivers geometry
    sx = 0
    sy = 0
   
    Hsource    = [sx, sy, -height]
    Hreceivers = [offsets, offsets*0, -height]

    Vsource    = [sx, sy, -height]
    Vreceivers = [offsets, offsets*0, -height]

    Psource    = [sx, sy, -height]
    Preceivers = [offsets+0.1, offsets*0, -height]
    
    res =[1e6, 1/sigma1, 1/sigma2]
    depth=[0, thicks1]

    # Compute fields
    HCP_Hs = empymod.dipole(Hsource, Hreceivers, depth, res, freq, 
                            ab=66, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
#    VCP_Hs = empymod.dipole(Vsource, Vreceivers, depth, res, freq,
#                            ab =55, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hs = empymod.dipole(Psource, Preceivers, depth, res, freq, 
                            ab=46, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 

    HCP_Hp = empymod.dipole(Hsource, Hreceivers, depth=[], res=[1e6], freqtime=freq,
                          ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
#    VCP_Hp = empymod.dipole(Vsource, Vreceivers, depth=[], res=[1e6], freqtime=freq,
#                          ab=55, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hp = empymod.dipole(Psource, Preceivers, depth=[], res=[1e6], freqtime=freq, 
                            ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
#
    Q_HCP = (HCP_Hs/HCP_Hp).imag.amp() 
#    Q_VCP = (VCP_Hs/VCP_Hp).imag.amp() 
    Q_PRP = (PRP_Hs/PRP_Hp).imag.amp() 
    
    P_HCP = (HCP_Hs/HCP_Hp).real.amp() 
#    P_VCP = (VCP_Hs/VCP_Hp).real.amp() 
    P_PRP = (PRP_Hs/PRP_Hp).real.amp() 
    
    return np.hstack((Q_HCP, Q_PRP, P_HCP, P_PRP))


def EMforward3lay(sigma1, sigma2, sigma3, thicks1, thicks2, offsets = np.array([2, 4, 8]), height=0, freq=9000):
    """ This function performs the forward simulation of a the measurements in a
    low frequency electromagnetic induction device assuming a 3 layered earth
    using HCP, VCP and PRP geometries 
    
    Parameters:
    - sigma1: Conductivity of the first layer [S/m]
    - sigma2: Conductivity of the second layer [S/m]
    - sigma3: Conductivity of the third layer [S/m]
    - thicks1: Thickness of the first layer [m]
    - thicks2: Thickness of the second layer [m]
    - offsets: separations of the transmitter and receiver coils [m]
    - height: height of the device with respect to ground [m]
    - freq: frequency of device [Hz]
    
    Returns:
    Array [Q_HCP, Q_VCP, Q_PRP, P_HCP, P_VCP, P_PRP]
    - Q_HCP : Quadrature Horizontal Coplanar
    - Q_VCP : Quadrature Vertical Coplanar
    - Q_PRP : Quadrature Perpendicular
    - P_HCP : In-Phase Horizontal Coplanar
    - P_VCP : In-Phase Vertical Coplanar
    - P_PRP : In-Phase Perpendicular
    """  
   
    time.sleep(0.01)
    
    # Source and receivers geometry
    sx = 0
    sy = 0
   
    Hsource    = [sx, sy, -height]
    Hreceivers = [offsets, offsets*0, -height]

    Vsource    = [sx, sy, -height]
    Vreceivers = [offsets, offsets*0, -height]

    Psource    = [sx, sy, -height]
    Preceivers = [offsets+0.1, offsets*0, -height]
    
    res =[1e6, 1/sigma1, 1/sigma2, 1/sigma3]
    depth=[0, thicks1, thicks1+thicks2]

    # Compute fields
    HCP_Hs = empymod.dipole(Hsource, Hreceivers, depth, res, freq, 
                            ab=66, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
    VCP_Hs = empymod.dipole(Vsource, Vreceivers, depth, res, freq,
                            ab =55, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hs = empymod.dipole(Psource, Preceivers, depth, res, freq, 
                            ab=46, xdirect=None, verb=0)*(2j * np.pi *freq * mu_0) 

    HCP_Hp = empymod.dipole(Hsource, Hreceivers, depth=[], res=[1e6], freqtime=freq,
                          ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
    VCP_Hp = empymod.dipole(Vsource, Vreceivers, depth=[], res=[1e6], freqtime=freq,
                          ab=55, verb=0)*(2j * np.pi *freq * mu_0) 
    PRP_Hp = empymod.dipole(Psource, Preceivers, depth=[], res=[1e6], freqtime=freq, 
                            ab=66, verb=0)*(2j * np.pi *freq * mu_0) 
#
    Q_HCP = (HCP_Hs/HCP_Hp).imag.amp() 
    Q_VCP = (VCP_Hs/VCP_Hp).imag.amp() 
    Q_PRP = (PRP_Hs/PRP_Hp).imag.amp() 
    
    P_HCP = (HCP_Hs/HCP_Hp).real.amp() 
    P_VCP = (VCP_Hs/VCP_Hp).real.amp() 
    P_PRP = (PRP_Hs/PRP_Hp).real.amp() 
    
    return np.hstack((Q_HCP, Q_VCP, Q_PRP, P_HCP, P_VCP, P_PRP))

def GlobalSearch(Database, Data, conds, thicks, nsl=51):
    """ This function searches through the lookup table database
    for the best data fit, and then finds the corresponding model 
    
    Parameters:
    1. Database: Lookup table
    2. Data: measurement for one position 
    3. conds: Conductivities sampled in the lookup table
    4. thicks: thicknesses sampled in the lookup table
    5. nsl: number of samples
    
    Returns: 2 layered model estimated through best data fit
    model = [sigma_1, sigma_2, h_1]
    
    Units:
    sigma_1 [S/m]
    sigma_2 [S/m]
    h_1 [m]
    """
    
    err = 1
    indx = 0
    
    # Search best data fit
    for i in range(np.shape(Database)[0]):
        nZdiff = (Database[i] - Data) **2 / (((Database[i] + Data)/2)**2)
        merr = np.log10(np.sqrt(np.sum(nZdiff)/len(Data)))
        if merr < err:
            indx = i
            err = merr.copy()
            
    # Find corresponding model
    for i in range(len(conds)):
        for j in range(len(conds)):
            for k in range(len(thicks)):
                idx = k + j*nsl + i*nsl**2
                if indx == idx:
                    model = np.array([conds[i], conds[j], thicks[k]])
    
    return model

def ErrorSpace(Database, Data, max_error, conds, thicks, nsl=51):
    err = []
    models_below_err = []
    for d in range(np.shape(Database)[0]):
        nZdiff = (Database[d] - Data) **2 / (((Database[d] + Data)/2)**2)
        merr = (np.sqrt(np.sum(nZdiff)/len(Data)))
        
        if merr < max_error:
            err.append(merr)
            indx = d
            for i in range(len(conds)):
                for j in range(len(conds)):
                    for k in range(len(thicks)):
                        idx = k + j*nsl + i*nsl**2
                        if indx == idx:
                            model = np.array([conds[i], conds[j], thicks[k]])
                            models_below_err.append(model)
    return np.array(err), np.array(models_below_err)

# Define cummulative sensitivity functions
# to calculate relative influence of layer

# z is the normalized depth with respect to coil separation (depth/coil sep)

def R_PRP(z):
    R = 2*z / (np.sqrt(4 * z**2 + 1))
    return R
    
def R_HCP(z):
    R = 1- 1/(np.sqrt(4 * z**2 + 1))
    return R
    
def R_VCP(z):
    R = 1- (np.sqrt(4 * z**2 + 1) - 2*z)
    return R 

def Sigma_from_Q(Q, s, freq=9000, mu = mu_0):
    """ Calculates sigma apparent from Q """
    sigma = 4 * Q / ((2 *np.pi * freq) * mu_0 * s**2 )
    return sigma

# Calculate induction number

def beta(s, sigma, freq=9000, mu_0 = mu_0):
    """ calculates induction number for a certain conductivity and coil separation s
    beta = s/ delta
    s coil separation in meters
    sigma in Siemens / m """
    skin_depth = np.sqrt(2 / (2 * np.pi * freq * mu_0 * sigma))
    return s / skin_depth 

def Q_from_Sigma(sigma, s, freq=9000, mu_0=mu_0):
    """ Function that back transforms Sigma_app to Quadrature values
    using the LIN approximation function 
    
    Parameters: 
    1. sigma: apparent conductivity [S/m]
    2. s: coil offset [m]
    
    Returns:
    Q : quadrature values
    """
    Q = sigma * (2 *np.pi * freq) * mu_0 * s**2 /4
    return Q

def LINforward2lay(sigma1, sigma2, thick1, height=0):
    """ Forward function that recreates values from DUALEM842 instrument
    using the Low induction number approximation (McNeill 1980)"""
    
    offsets = np.array([2, 4, 8]) # in meters
    freq = 9000

    Q_HCP = []
    Q_VCP = []
    Q_PRP = []

    for coil in offsets: 
        sigma_a_HCP = ((1/1e6) * (1- R_HCP(height/coil)) + 
                          (sigma1 * (R_HCP(thick1/coil) - R_HCP(height/coil))) + 
                          sigma2 *(R_HCP(thick1/coil)))

        sigma_a_VCP = ((1/1e6) * (1-R_VCP(height/coil)) + 
                          (sigma1 * (R_VCP(thick1/coil) - R_VCP(height/coil))) + 
                          sigma2 *(R_VCP(thick1/coil)))

        sigma_a_PRP = ((1/1e6) * (1-R_PRP(height/(coil+.1))) + 
                          (sigma1 * (R_PRP(thick1/(coil+.1)) - R_PRP(height/(coil+.1)))) + 
                          sigma2 *(R_PRP(thick1/(coil+.1))))


        Q_HCP.append(Q_from_Sigma(sigma_a_HCP, coil))
        Q_VCP.append(Q_from_Sigma(sigma_a_VCP, coil))
        Q_PRP.append(Q_from_Sigma(sigma_a_PRP, coil+.1))

    return np.hstack((Q_HCP, Q_VCP, Q_PRP))
            
def ErrorSpace3Lay(m_est_pos, data_true_pos, conds, thicks, max_err=0.12):
    # Evaluate only conductivity and thickness of middle layer
    err = []
    models_below_err = []
    
    for c2 in conds:
        for t1 in thicks:
            for t2 in thicks:
                m = [m_est_pos[0], c2, m_est_pos[2], t1, t2]
                data_est_pos = EMforward3lay(m_est_pos[0], c2, m_est_pos[2], t1, t2) 
                nZdiff = (data_true_pos - data_est_pos) **2 / (((data_true_pos + data_est_pos)/2)**2)
                merr = (np.sqrt(np.sum(nZdiff)/len(data_true_pos)))

                if merr < max_err:
                    err.append(merr)       
                    models_below_err.append(m)
                    
    return np.array(err), np.array(models_below_err) 

def distance(lat1, lat2, lon1, lon2):
    """ Function to calculate the horizontal distance between to 
    coordinates in degrees """
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)  
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c # distance in km
    return distance*1000 # distance in m

def GlobalSearch_FS(Database, Data):
    """ This function searches through the lookup table database
    for the best data fit, and then finds the corresponding model 
    
    Modified for faster search
    
    Parameters:
    1. Database: Lookup table
    2. Data: measurement for one position 
    
    Returns: [err, indx]
    
    err = error of best fit in lookup table
    indx = index of best fit in lookup table
    """

    # Evaluate for min error
    nZdiff = (Database[:] - Data)**2/(((Database[:]+ Data)/2)**2)
    merr = np.log10(np.sqrt(np.sum(nZdiff, axis=1))/len(Data))
    err = np.min(merr)
    indx = np.argmin(merr) 
    
    return np.array([err, indx])
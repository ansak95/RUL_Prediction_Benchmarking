"""
Generating crack propagation data
"""
from pandas.io.json import json_normalize

from numba import jit
import argparse
import configparser
import json
import pickle

import numpy as np
import scipy.stats

import crack
import pandas as pd

from scipy.stats import bernoulli
from pandas.io.json import json_normalize

@jit
def gen_param_sample(a_0_mean, a_0_std, C_mean, C_std, m_mean, m_std):
    """
    Generate joint sample of crack parameters :math:`a_0`, :math:`C`,
    :math:`m`.

    Parameters
    ----------
    a_0_mean : float
        Mean of initial half crack length
    a_0_std : float
        Standard deviation of initial half crack length
    C_mean : float
        Mean of crack parameter :math:`C`
    C_std : float
        Std dev of crack parameter :math:`C`    
    log_C_std : float
        Standard deviation on the natural logarithm of crack parameter
        :math:`C` 
    m_mean : float
        Mean of crack parameter :math:`m`
    m_std : float
        Standard deviation onf crack parameter :math:`m`

    Returns
    -------
    a_0, C, m
    """
    # Truncated Gaussian law for the initial crack length
    a_0_scaled_lower = 0#(0 - a_0_mean)/a_0_std # Scale lower limit (0) appropriately for scipy; upper limit is infinite at any scale
    a_0 = scipy.stats.truncnorm.rvs(a_0_scaled_lower, np.inf, a_0_mean, a_0_std) # Perform the actual sample

    log_C_mean = np.log((C_mean**2)/np.sqrt(C_mean**2 + C_std**2))#np.log(C_mean)
    log_C_std = np.log(1+ (C_std**2)/(C_mean**2))
    
    C = 0
    rho = -0.996
    m = -1

    cov = np.array(((log_C_std**2, rho * log_C_std * m_std), (rho * log_C_std * m_std, m_std**2)))
    while m < 0:
        params = scipy.stats.multivariate_normal.rvs((log_C_mean, m_mean), cov)
        
        C = np.exp(params[0])
        m = params[1]

    return a_0, C, m

def gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu):
    """
    Generate strain value for gauge position and crack length

    ``crack.strain`` provides the full strain state, but strain gauges can
    only deliver a strain measurement in one direction. This routine calculates
    the strain value "seen" by a unidirectional strain gauge from the full
    strain state.

    Parameters
    ----------
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    a : float
         Half crack length
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    
    Returns
    -------
    The strain value for the given gauge positions the gauge "sees".
    """
    epsilon_11, epsilon_22, epsilon_12, epsilon_33  = crack.strain(np.abs(x_gauge), y_gauge, a, delta_sigma, E, nu)

    # Calculate the unidirectional strain the gauge "sees" ( and store it
    epsilon_gauges = epsilon_11 * np.cos(theta_gauge)**2 + epsilon_22 * np.sin(theta_gauge)**2 + 2* epsilon_12 * np.cos(theta_gauge) * np.sin(theta_gauge)

    return epsilon_gauges, epsilon_11, epsilon_22, epsilon_12

@jit
def gen_dataset(type_data, percentage_distortion, x_gauge, y_gauge, theta_gauge, delta_sigma, E, nu, a_0_mean, a_0_std, a_crit, C_mean, log_C_std, m_mean, m_std, n_samples, thinning):
    """
    Generate a dataset for crack propagation measurement data.

    Parameters
    ----------
    type_data : string
    'train', 'val' or 'test' dataset
    a0_var : string
    'fixed' or 'variable'
    alpha : int
    size of the confidence interval, 99 or 95
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    a_0_mean : float
        Mean of initial half crack length
    a_0_std : float
        Standard deviation of initial half crack length
    C_mean : float
        Mean of crack parameter :math:`C`
    log_C_std : float
        Standard deviation on the natural logarithm of crack parameter
        :math:`C` 
    m_mean : float
        Mean of crack parameter :math:`m`
    m_std : float
        Standard deviation onf crack parameter :math:`m`
    n_samples : int
        Number of samples
    thinning : int
        Provide samples for every ``thinning`` cycle

    Returns
    -------
    Dictionary of crack propagation sequence datasets, each containing
    C
      :math:`C` parameter of Paris' law, used for the generation of the
      sequence
    m
      :math:`m` parameter of Paris' law , used for the generation of
      the sequence
    a_0
      Half crack length at the start of the crack propagation
    x_gauges
      :math:`x` positions of the strain gauges
    y_gauges
      :math:`y` positions of the strain gauges
    crack_lengths
      Crack lengths for every ``thinning`` cycles
    initial_strains
      Strains at initial crack length
    strains_relative
      Incremental changes of the strain at each ``thinning`` cycles, or rate of change
    """
    # Sample "experimental settings"
    sequence_datasets = []

    
    for i in range(n_samples):
        
        # Sample parameters for individual sequence
        a_0, C, m = gen_param_sample(a_0_mean, a_0_std, C_mean, log_C_std, m_mean, m_std)
        
            
        k,crack_lengths_numpy, strains_numpy, sigma, epsilon, displacements = gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning)
        
        initial_strains = strains_numpy[0].tolist()
#         strains_relative = ((strains_numpy[1:] - strains_numpy[:-1])/strains_numpy[:-1])
#         diff_relatives = np.diff(strains_numpy, axis=0, n=7)

#         print(strains_numpy.shape)
#         print(np.std(strains_numpy, axis = 1))

#         https://stackoverflow.com/questions/64074698/how-to-add-5-gaussian-noise-to-the-signal-data

        # Create a dataset and store in global object
        print('Created dataset no. {0: d} with {1: d} cycles'.format(i + 1, (strains_numpy.shape[0] - 1)*thinning)) 
        
        dataset = {'C': C, 'm': m, 'a_0': a_0, 'x_gauges': (x_gauge), 'y_gauges': (y_gauge), 
                   'crack_lengths': list(crack_lengths_numpy), 'nb_cycles' : k, 'Nb_measures' : k//thinning + 1, 'strains' : strains_numpy}
        
#         dataset = {'C': C, 'm': m, 'a_0': a_0, 'x_gauges': (x_gauge), 'y_gauges': (y_gauge), 'sigma': list(sigma),
#                    'epsilon': list(epsilon), 'displacements': list(displacements),
#                    'crack_lengths': list(crack_lengths_numpy), 'nb_cycles' : k, 'Nb_measures' : k//thinning + 1, 'initial_strains': initial_strains, 'strains' : strains_numpy}
        sequence_datasets.append(dataset)
        
    return pd.DataFrame.from_dict(json_normalize(sequence_datasets), orient='columns')



@jit
def gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning):
    """
    Generate one crack propagation sequence from parameters

    Parameters
    ----------
    a_0 : float
         Initial half crack length
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    thinning : int
         Record strain every ``thinning`` cycles

    Returns
    -------
    Two arrays containing the crack length and strains at every ``thinning``th cycle
    """
    
    sigma_inf = delta_sigma - 0
    
    # Calculate initial strains & stress & displacements
    epsilon_gauges, epsilon_11, epsilon_22, epsilon_12 = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a_0, delta_sigma, E, nu) 
    sigma_11, sigma_22, sigma_12 = crack.stress(x_gauge, y_gauge, a_0, sigma_inf)
    u,v = crack.displacement(x_gauge, y_gauge, a_0, sigma_inf, E, nu, state='plane stress')

    # Initialize sequence generation loop with initial crack length, strains & stress & displacements
    crack_lengths = [a_0]
    
    sigma = [(sigma_11,sigma_22,sigma_12)]
    epsilon = [(epsilon_11,epsilon_22,epsilon_12)]
#     sigma_xx = [sigma_11]
#     sigma_yy = [sigma_22]
#     sigma_xy = [sigma_12]
    
#     epsilon_xx = [epsilon_11]
#     epsilon_yy = [epsilon_22]
#     epsilon_xy = [gamma_12]
    strains = [epsilon_gauges]
    
    displacements = [(u,v)]
    
#     dis_u = [u]
#     dis_v = [v]

    k = 0
    a = a_0
    
#     strains_arr = [epsilon_gauges]
    
    # Propagate until the crack is a_crit-mm long, at which point we consider the sample to be broken
    while a < a_crit:
        # next cycle
        k = k + 1
        
        # Calculate the crack length and store it
        a = crack.length_paris_law(k, delta_sigma, a_0, C, m)
        
        
        
        if (k % thinning == 0):
            
            crack_lengths.append(a)
            
            epsilon_gauges, epsilon_11, epsilon_22, epsilon_12 = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu)
            sigma_11, sigma_22, sigma_12 = crack.stress(x_gauge, y_gauge, a, sigma_inf)
            u,v = crack.displacement(x_gauge, y_gauge, a, sigma_inf, E, nu, state='plane stress')
            
#             print(sigma_xx)
#             sigma_xx= sigma_xx+sigma_11
#             print(sigma_xx)#.append(list(sigma_11))
#             sigma_yy.append(list(sigma_22))
#             sigma_xy.append(list(sigma_12))


            strains.append((epsilon_gauges))

#             epsilon_xx.append(list(epsilon_11))
#             epsilon_yy.append(list(epsilon_22))
#             epsilon_xy.append(list(gamma_12))
#             print(strains)
#             print(epsilon)
#             print((epsilon_11, epsilon_22, gamma_12))
            epsilon.append(list([epsilon_11, epsilon_22, epsilon_12]))
            sigma.append(list([sigma_11, sigma_22, sigma_12]))
            displacements.append(list([u,v]))
#             dis_v.append(list(v))            
            
            #features
#             mean_strains = strains_arr.mean()
#             min_strains = strains_arr.max()
#             max_strains = strains_arr.min()
#             std_strains = strains_arr.std()
#         else :
            
#             #just store strain values
#             epsilon_gauges, _, _, _ = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu)
#             strains_arr.append((epsilon_gauges))


    return k,np.array(crack_lengths), np.array(strains), np.array(sigma),np.array(epsilon), np.array(displacements)

@jit
def add_noise(db, percentage_distortion) :
    temp = pd.DataFrame()
    seq = []
    for i in range(db.shape[0]) :        
        #add noise
        X = db.loc[i].strains[:,:] 
        nu = np.random.normal(0,np.std(np.std(X, axis = 0), axis = 0),(X.shape[0],X.shape[1])) * percentage_distortion
        Xprime = X + nu
        dataset = {'noisy_strains' : Xprime}
        seq.append(dataset)
    
    
    temp = pd.DataFrame.from_dict(json_normalize(seq), orient='columns')
    db['noisy_strains'] = temp.noisy_strains 
    return db
@jit
def set_threshold_variate(db, type_data, lower_boundary, upper_boundary) :
   
    db['t_star'] = 0 #t* which is actually the size of the sequence available (especially for test set, structures before failure)
    db['t_star'] = (np.random.uniform(low=lower_boundary, high=upper_boundary, size=(db.shape[0],)) * db.nb_cycles.values).astype(int)
#     print(db.)
    if (type_data == 'raw_train') :
        db.loc[(db.until_failure==1) , 't_star'] = db.loc[(db.until_failure==1), 'nb_cycles'] #limite limite...

    while db.loc[(db.t_star<=5000)].shape[0] > 0 :
    
        db.loc[(db.t_star<=5000), 't_star'] = (np.random.uniform(low=lower_boundary, high=upper_boundary, size=(db.loc[(db.t_star<=5000)].shape[0],)) * db.loc[(db.t_star<=5000)].nb_cycles.values).astype(int) #limite limite...
    
    return db
@jit
def preprocess_raw_data(db, type_data, p, percentage_distortion, lb = 0.33, ub = 0.95) :
    
    """
    type_data : string
    'raw_train', train', 'val' or 'test' dataset
    """
    
    if type_data == 'raw_train' : 
        db['until_failure'] = bernoulli.rvs(p = p,size = db.shape[0]) #p parameter
    
    if type_data == 'train' : 
        db['until_failure'] = bernoulli.rvs(p = 1,size = db.shape[0]) #p parameter
    
    db = add_noise(db, percentage_distortion= percentage_distortion)
    db = set_threshold_variate(db, type_data, lower_boundary = lb, upper_boundary = ub)
    return db
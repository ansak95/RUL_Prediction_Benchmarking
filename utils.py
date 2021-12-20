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



from numba import jit
import timeit
import time

@jit
def build_dataset(data, cols,   nb_gauges = 3, stop_tstar = False, thinning = 500) :
    """
    For the moment only 3 gauges
    
    Parameters
    ----------
    data : dataset, unstructured dataset
    cols : string array, i.e. columns considered for the structure dataframe 
    nb_gauges : int, number on considered gauges
    stop_tstar : bool, True if the measures are collected until a fixed time t* 
    (especially for testing set, where the sequences are incomplete)
    
    thinning : int

    Returns
    -------
    Structured dataframe
    """

    tmp = 0
    
    db = pd.DataFrame(None) #intitialize the dataframe
    nb_rows = 0
    len_prev = 0 #variable used to concatenate multiple sub-dataframes (a dataframe per structure) into a large dataframe (name db here)
    
    
    start = timeit.default_timer()
    t0 = time.time()
    
    
    for i in range(data.shape[0]) :
 
        len_max = data.ix[i,'Nb_measures'] #nb of measures for structure i
        nb_cycles = data.loc[i].nb_cycles #nb of cycles for structure i
        nb_measures = data.ix[i,'Nb_measures']
        
        #initialize a sub-dataframe for structure i
        temp = pd.DataFrame(np.zeros((len_max,len(cols))), columns =cols, dtype='float').astype({"cycle": int, "ID": int, "RUL": int}) 
        #append the sub-dataframe to the considered dataframe
        db = db.append(temp, ignore_index=True)
        
        #fill the empty dataframe
        db.ix[len_prev:len_prev+len_max,'ID'] = i + 1   
        db.ix[len_prev:len_prev+len_max,'cycle'] = thinning*np.linspace(0, len_max-1, len_max, dtype='int')   
        #raw measures, i.e. the input columns
        for k in range(nb_gauges) : 
            db.ix[len_prev:len_prev+len_max,'gauge' + str(k+1)] = data.ix[i].strains[0:len_max,k]
        #create and fill the output column
        db.ix[len_prev:len_prev+len_max,'RUL'] = np.array(range(nb_cycles,nb_cycles%thinning - thinning, -thinning)) 
        
        
        len_prev = len_prev + len_max #variable used to concatenate multiple sub-dataframes (a dataframe per structure, named temp here) into a large dataframe (named db here)

        
    stop = timeit.default_timer()
    print('Time: ', (stop - start), 's. Sample', i+1)
    
    
    
    if  stop_tstar == True : #if sequences incomplete
        
        for i in range(data.shape[0]) :
            temp_t = data.loc[i].t_star
            ind = db[(db.ID == i+1) & (db.cycle > temp_t)].index #for structure i, select in the sub-dataframe the rows of measures after time t*
            db.drop(ind, inplace = True) #drop the rows

    
    return db



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

    log_C_mean = np.log((C_mean**2)/np.sqrt(C_mean**2 + C_std**2))
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

    # Calculate the unidirectional strain gauge and store it
    epsilon_gauges = epsilon_11 * np.cos(theta_gauge)**2 + epsilon_22 * np.sin(theta_gauge)**2 + 2* epsilon_12 * np.cos(theta_gauge) * np.sin(theta_gauge)

    return epsilon_gauges, epsilon_11, epsilon_22, epsilon_12

@jit
def gen_dataset(type_data, x_gauge, y_gauge, theta_gauge, delta_sigma, E, nu, a_0_mean, a_0_std, a_crit, C_mean, C_std, m_mean, m_std, n_samples, thinning):
    """
    Generate a dataset for crack propagation measurement data.

    Parameters
    ----------
    type_data : string
    'train', 'val' or 'test' dataset
    
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
    C_std : float
        Std dev of crack parameter :math:`C` 
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
    """
    # Sample "experimental settings"
    sequence_datasets = []

    
    for i in range(n_samples):
        
        # Sample parameters for individual sequence
        a_0, C, m = gen_param_sample(a_0_mean, a_0_std, C_mean, C_std, m_mean, m_std)
        
            
        #k,crack_lengths_numpy, strains_numpy, sigma, epsilon, displacements = gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning)
        
        k,crack_lengths_numpy, strains_numpy = gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning)
        
        initial_strains = strains_numpy[0].tolist()

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
    #sigma_11, sigma_22, sigma_12 = crack.stress(x_gauge, y_gauge, a_0, sigma_inf)
    #u,v = crack.displacement(x_gauge, y_gauge, a_0, sigma_inf, E, nu, state='plane stress')

    # Initialize sequence generation loop with initial crack length and strains
    crack_lengths = [a_0]
    strains = [epsilon_gauges]
#     sigma = [(sigma_11,sigma_22,sigma_12)]
#     epsilon = [(epsilon_11,epsilon_22,epsilon_12)]

    
    
#     displacements = [(u,v)]
    

    k = 0
    a = a_0
 

    # Propagate until the crack is a_crit-mm long, at which point we consider the sample to be broken
    while a < a_crit:
        # next cycle
        k = k + thinning
        
        # Calculate the crack length and store it
        a = crack.length_paris_law(k, delta_sigma, a_0, C, m)
        crack_lengths.append(a)
        
        # Calculate the strains and store it
        epsilon_gauges, epsilon_11, epsilon_22, epsilon_12 = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu)
        strains.append((epsilon_gauges))
        
#         sigma_11, sigma_22, sigma_12 = crack.stress(x_gauge, y_gauge, a, sigma_inf)
#         u,v = crack.displacement(x_gauge, y_gauge, a, sigma_inf, E, nu, state='plane stress')


        
        
        #epsilon.append(list([epsilon_11, epsilon_22, epsilon_12]))
        #sigma.append(list([sigma_11, sigma_22, sigma_12]))
        #displacements.append(list([u,v]))


    return k,np.array(crack_lengths), np.array(strains)#, np.array(sigma),np.array(epsilon), np.array(displacements)


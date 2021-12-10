"""
Determine stress, strain and displacement for a finite crack of
length :math:`2a` under mode I loading.
"""

import numpy as np

def phi(z, a, sigma_inf):
    """
    Complex stress function for finite crack in infinite plate.
    See [Zehnder2012]_, p. 23, Eq. (2.77).

    Parameters
    ----------
    z : array_like, complex
        Complex coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.

    Returns
    -------
    Stress function value at :math:`z`.
    """
    return sigma_inf * np.sqrt(z**2 - a**2) - sigma_inf * z

def phi_prime(z, a, sigma_inf):
    """
    First derivative of stress function for finite crack in infinite
    plate. See [Zehnder2012]_, p. 23, Eq. (2.76).

    Parameters
    ----------
    z : array_like, complex
        Complex coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.

    Returns
    -------
    First derivative of the stress function at :math:`z`.
    """
    return sigma_inf * z / np.sqrt(z**2 - a**2) - sigma_inf

def phi_second(z, a, sigma_inf):
    """
    Second derivative of stress function for finite crack in infinite
    plate. See [Zehnder2012]_, p. 23, Eq. (2.76).

    Parameters
    ----------
    z : array_like, complex
        Complex coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.

    Returns
    -------
    Second derivative of the stress function at $z$.
    """
    return sigma_inf/np.sqrt(z**2 - a**2) - sigma_inf * z**2/((z**2 - a**2)**(1.5))

def displacement(x, y, a, sigma_inf, E, nu, state='plane stress'):
    """
    Calculate displacement. See [Zehnder2012]_, p. 22, Eq. (2.70).

    Parameters
    ----------
    x : array_like
        x coordinate.
    y : array_like
        y coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.
    E : float
        Young's modulus
    nu : float
        Poisson's ratio.
    state : str
        Assumed state. Either 'plane stress' (default) or 'plane strain'

    Returns
    -------
    Displacements in direction of $x$ ($u$) and $y$ ($v$)
    """
    mu = E/(2 * (1 + nu))
    z = np.array(x) + [1j*y_unit for y_unit in y]#x+1j*y#np.array(x) + [1j*y_unit for y_unit in y]#1j*y
    
    kappa = (3 - nu)/(1 + nu)
    
    if state == 'plane strain':
        kappa = 3 - 4*nu
    
    u = 1/(2 * mu) * ((kappa - 1)/2 * np.real(phi(z, a, sigma_inf)) - y * np.imag(phi_prime(z, a, sigma_inf)))
    v = 1/(2 * mu) * ((kappa + 1)/2 * np.imag(phi(z, a, sigma_inf)) - y * np.real(phi_prime(z, a, sigma_inf)))

    return u, v

def stress(x, y, a, sigma_inf, state='plane stress'):
    """
    Calculate the stresses. See [Zehnder2012]_, p. 22, Eq. (2.69).

    Parameters
    ----------
    x : array_like
        x coordinate.
    y : array_like
        y coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.
    state : str
        Either 'plane stress' or 'plane strain'.

    Returns
    -------
    Stresses sigma_11, sigma_22, tau_12, sigma_33 (if present) (in that order).
    """
    z = np.array(x) + [1j*y_unit for y_unit in y] ##x+1j*y#np.array(x) + [1j*y_unit for y_unit in y]#1j*y

    if state == 'plane stress':
        sigma_11 = np.real(phi_prime(z, a, sigma_inf)) - y * np.imag(phi_second(z, a, sigma_inf))
        sigma_22 = np.real(phi_prime(z, a, sigma_inf))  + y * np.imag(phi_second(z, a, sigma_inf)) + sigma_inf
        tau_12 = -np.array(y) * np.real(phi_second(z, a, sigma_inf))

        return sigma_11, sigma_22, tau_12

    else:
        sigma_11 = np.real(phi_prime(z, a, sigma_inf)) - y * np.imag(phi_second(z, a, sigma_inf))
        sigma_22 = np.real(phi_prime(z, a, sigma_inf))  + y * np.imag(phi_second(z, a, sigma_inf)) + sigma_inf
        tau_12 = -np.array(y) * np.real(phi_second(z, a, sigma_inf))

        epsilon_11 = (1 + nu)/E * ((1 - nu) * sigma_11 - nu * sigma_22)
        epsilon_22 = (1 + nu)/E * (-nu * sigma_11 + (1 - nu) * sigma_22)
        
        sigma_33 = E * nu * (epsilon_11 + epsilon_22)/((1 - 2*nu) * (1 + nu))
        return sigma_11, sigma_22, tau_12, sigma_33
        

def strain(x, y, a, sigma_inf, E, nu, state='plane stress'):
    """
    Calculate strains.

    Parameters
    ----------
    x : array_like
        x coordinate.
    y : array_like
        y coordinate.
    a : float
        Half crack length.
    sigma_inf : float
        Load in the far field.
    E : float
        Young's modulus
    nu : float
        Poisson's ratio.
    state : str
        Assumed state. Either 'plane stress' (default) or 'plane strain'

    Returns
    -------
    Strains epsilon_11, epsilon_22, gamma_12, (epsilon_33 if present).
    """

    sigma_11, sigma_22, sigma_12 = stress(x, y, a, sigma_inf)

    if state == 'plane strain':
        epsilon_11 = (1 + nu)/E * ((1 - nu) * sigma_11 - nu * sigma_22)
        epsilon_22 = (1 + nu)/E * (-nu * sigma_11 + (1 - nu) * sigma_22)
        epsilon_12 = (1 + nu)/E * sigma_12
        return epsilon_11, epsilon_22, epsilon_12
    else:
        epsilon_11 = 1/E * (sigma_11 - nu * sigma_22)
        epsilon_22 = 1/E * (-nu * sigma_11 + sigma_22)
        epsilon_12 = (1 + nu)/E * sigma_12
        epsilon_33 = -nu/E * (sigma_11 + sigma_22)
        
        return epsilon_11, epsilon_22, epsilon_12, epsilon_33

def length_paris_law(k, delta_sigma, a_0, C, m):
    """
    Calculate crack length after :math:`k` cycles after
    Paris law.

    Parameters
    ----------
    k : int
        Number of cycles
    delta_sigma : float
        Stress delta
    a_0 : float
        Initial crack length
    C : float
        Material parameter
    m : float
        Paris law exponent

    Returns
    -------
    Crack length after :math:`k` cycles
    """

    a_k = (k * C * (1 - m/2)*(delta_sigma*1e-6 * np.sqrt(np.pi))**m + a_0**(1 - m/2))**(2/(2 - m))
    return a_k


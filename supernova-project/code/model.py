import math
import pandas as pd
import numpy as np
from sympy import symbols, diff

# intermediary eta function
def eta(a, omega_m):
    """
    Calculate the eta function for a given scale factor and matter density parameter.
    Parameters:
    a (float): Scale factor.
    omega_m (float): Matter density parameter.
    Returns:
    float: Value of the eta function.
    """
    s = ((1 - omega_m) / omega_m) ** (1.0 / 3)
    return 2 * omega_m ** (-0.5) * (1 / (a ** 4) - 0.1540 * s / (a ** 3)
                                    + 0.4304 * s ** 2 / (a ** 2)
                                    + 0.19097 * s ** 3 / a
                                    + 0.066941 * s ** 4) ** (-1.0 / 8)

# function to calculate distance modulus from redshifts
def dist_mod_func(redshift, omega_m, hubble_const):
    """
    Calculate the distance modulus for a given redshift, matter density parameter, and Hubble constant.
    Parameters:
    redshift (float): Redshift.
    omega_m (float): Matter density parameter.
    hubble_const (float): Hubble constant.
    Returns:
    float: Distance modulus.
    """
    DL = 3000 * (1 + redshift) * (eta(1, omega_m) - eta(1 / (1 + redshift), omega_m))
    return 25 - 5 * math.log10(hubble_const) + 5 * math.log10(DL)

# function to generate theoretical curves using the distance modulus function
def supernova_curves(omega_m_values, hubble_values):
    """
    Generate theoretical curves for distance modulus as a function of redshift.
    Parameters:
    omega_m_values (list): List of matter density parameters.
    hubble_values (list): List of Hubble constants.
    Returns:
    pd.DataFrame: DataFrame containing redshift and corresponding distance modulus values.
    """
    zs = np.linspace(0.01, 2, 100)  # redshift range
    theoretical_curves = {
        f"omega_m={omega_m}, h = {h}": [dist_mod_func(z, omega_m, h) for z in zs]
        for omega_m in omega_m_values for h in hubble_values
    }

    return pd.DataFrame({'redshift': zs, **theoretical_curves})

def simulate_supernovae(min_redshift, max_redshift, sample_size, omega_m_vals, hubble_vals, mean, rms):
    """
    Simulate multiple supernovae with redshifts in some range with added gaussian noise to their distance modulus i.e. 'brightness'.
    Simulate a range of scenarios for matter density and Hubble constant.
    Parameters:
    min_redshift (float): Minimum redshift.
    max_redshift (float): Maximum redshift.
    sample_size (int): Number of samples to generate.
    omega_m_vals (list): List of matter density parameters.
    hubble_vals (list): List of Hubble constants.
    mean (float): Mean of the normal distribution for noise.
    rms (float): Standard deviation of the normal distribution for noise.
    Returns:
    pd.DataFrame: DataFrame containing redshift and corresponding distance modulus values with noise.
    """
    zs_random=np.random.uniform(min_redshift, max_redshift, size=sample_size)

    simulated_data = {
        f"omega_m={omega_m}, h={h}": [dist_mod_func(z, omega_m, h)+np.random.normal(mean, rms) for z in zs_random]
        for omega_m in omega_m_vals for h in hubble_vals
    }

    return pd.DataFrame({'redshift': zs_random, **simulated_data})

def model_log_likelihood(mu_data, mu_theory, cov_data):
    cov_matrix=np.array(cov_data).reshape((len(mu_data), len(mu_data)))
    diff = mu_data - mu_theory
    chi = -0.5 * np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff)
    return chi

def logflatprior(x):
    return 0

def loggaussprior(mean, stdev):
    return lambda x: -0.5 * ((x - mean) / stdev) ** 2

def empirical_log_posterior(omega_m, h, sample_means, sample_cov_df):
    """
    Calculate the log posterior probability assuming a simple bivariate normal model with 
    the parameter means and covariance matrix according to the previous sampled theoretical model.

    Parameters
    ----------
    omega_m : float
        Omega_m parameter.
    h : float
        Hubble constant parameter.
    sample_means : pd.Series
        Sample means of the parameters.
    sample_cov_df : pd.DataFrame
        Sample covariance matrix of the parameters.
    Returns
    -------
    float
        Log posterior probability.
    """
    diff = (np.array([omega_m, h])- sample_means.to_numpy())
    return -0.5 * np.dot(np.dot(diff, np.linalg.inv(sample_cov_df.to_numpy())), diff)
import numpy as np
import pandas as pd
from typing import Callable
from functools import partial

def supernova_metropolis_hastings(mu_data, z_data, cov_data, 
                        omega_m_init:float, h_init:float, n_steps:int, 
                        omega_m_step_size:float, h_step_size:float, 
                        hubble_log_prior:Callable[[float], float], 
                        omega_m_log_prior:Callable[[float], float]):
    
    """
    Perform Metropolis-Hastings sampling to estimate the posterior distribution of supernova model parameters.
    Uses a anormal proposal distribution for the parameters.
    
    Parameters:
    mu_data (list): Observed distance modulus data.
    z_data (list): Redshift data.
    cov_data (list): Covariance data.
    """
    
    from model import dist_mod_func, model_log_likelihood

    omega_m = omega_m_init
    h = h_init 
    mu_theory = [dist_mod_func(z, omega_m, h) for z in z_data]
    log_post = model_log_likelihood(mu_data, mu_theory, cov_data)+hubble_log_prior(h)+omega_m_log_prior(omega_m)
    omega_m_chain = [omega_m]
    h_chain = [h]
    log_post_chain = [log_post]
    accepted = 0

    for _ in range(n_steps):
        omega_m_new = omega_m + np.random.normal(0, omega_m_step_size)
        h_new = h + np.random.normal(0, h_step_size)
        mu_theory = [dist_mod_func(z, omega_m_new, h_new) for z in z_data]
        log_post_new = model_log_likelihood(mu_data, mu_theory, cov_data)+hubble_log_prior(h_new)+omega_m_log_prior(omega_m_new)
        if np.log(np.random.rand()) < log_post_new - log_post: # condition for acceptance
            omega_m = omega_m_new
            h = h_new
            log_post = log_post_new
            accepted += 1

        omega_m_chain.append(omega_m)
        h_chain.append(h)
        log_post_chain.append(log_post)

    return pd.DataFrame({"omega_m":omega_m_chain, "hubble const":h_chain, "log post":log_post_chain}), accepted/n_steps

def normal_proposal_dist(params, step_sizes:np.ndarray):
    """
    Generate a proposal for the parameters using a multi-normal distribution with 
    variances given by step_sizes and mean 0.
    
    Parameters:
    params (list): Current parameter values.
    step_sizes (np.ndarray): Variances for the proposal distribution.
    
    Returns:
    np.ndarray: Proposed parameter values.
    """
    return np.random.normal(params, step_sizes)
    

def metropolis_hastings(model_log_liklihood, n_steps:int, sampling_method:pd.DataFrame):
    
    names=sampling_method["param_names"]
    params=sampling_method["params_init"]
    log_priors = sampling_method["param_log_priors"]
    proposal_dist = sampling_method["param_proposal_dist"]
    log_post = model_log_liklihood(params)+log_priors(params)
    log_post_chain = [log_post]
    params_chain = [params]
    accepted = 0

    for _ in range(n_steps):
        params_new = params+proposal_dist(params)
        log_post_new = model_log_liklihood(params_new)+log_priors(params_new)
        if np.log(np.random.rand()) < log_post_new - log_post: # condition for acceptance
            params = params_new
            log_post = log_post_new
            accepted += 1

        params_chain.append(params)
        log_post_chain.append(log_post)

    chains = pd.DataFrame({name:chain for name, chain in zip(names, np.array(params_chain).T)})
    chains["log post"] = log_post_chain

    return chains, accepted/n_steps

def simple_grad_log_posterior(sample_means, sample_cov_df):
    """
    Calculate the gradient of the log posterior probability of a bivariate normal approximation to the supernova model.

    This function uses symbolic differentiation to compute the gradient of the log posterior
    with respect to the parameters omega_m and h, and converts it into numerical functions.

    Parameters:
    sample_means (list): Sampled means of the model parameters.
    sample_cov_df (pd.DataFrame): Sampled covariance matrix across model parameters.
    Returns:
    Callable[[np.ndarray], np.ndarray]: A function that computes the numerical gradient of the log posterior
                                         with respect to omega_m and h.
    """

    from model import empirical_log_posterior
    from sympy import symbols, diff, lambdify

    omega_m, h = symbols("omega_m h")
    f = empirical_log_posterior(omega_m, h, sample_means, sample_cov_df)

    grad_log_post = [-diff(f, omega_m), -diff(f, h)] # symbolic gradients
    grad_log_post_func = lambdify((omega_m, h), grad_log_post, modules="numpy") # convert symbolic gradients to numerical functions

    return lambda x : np.array(grad_log_post_func(x[0], x[1])) # return a function that takes a single array of parameters

def hamiltonian_monte_carlo(model_log_posterior, grad_log_posterior, n_steps: int, param_df: pd.DataFrame, leapfrog_step_size: float, leapfrog_steps: int):
    """
    Perform Hamiltonian Monte Carlo (HMC) sampling.

    Parameters:
    model_log_likelihood (Callable): Function to compute the log-likelihood of the model.
    grad_log_likelihood (Callable): Function to compute the gradient of the log-likelihood.
    n_steps (int): Number of HMC steps to perform.
    sampling_method (pd.DataFrame): DataFrame containing parameter names, initial values, and priors.
    step_size (float): Step size for the leapfrog integrator.
    leapfrog_steps (int): Number of leapfrog steps to perform for each proposal.

    Returns:
    pd.DataFrame: Chains of sampled parameters and log-posterior values.
    float: Acceptance rate of the HMC sampler.
    """
    names = param_df["param_names"]
    params = np.array(param_df["params_init"])
    log_post = model_log_posterior(params)
    log_post_chain = [log_post]
    params_chain = [params]
    accepted = 0

    for _ in range(n_steps):
        momentum = np.random.normal(size=params.shape)
        hamiltonian = -log_post + 0.5 * np.sum(momentum**2)

        # perform leapfrog integration
        params_new = np.copy(params)
        grad = grad_log_posterior(params_new)
        momentum -= 0.5 * leapfrog_step_size * grad  # Half-step for momentum
        for _ in range(leapfrog_steps):
            params_new += leapfrog_step_size * momentum  # Full-step for position
            if _ != leapfrog_steps - 1:  # No momentum update on the last step
                momentum -= leapfrog_step_size * grad
        momentum -= 0.5 * leapfrog_step_size * grad  # Half-step for momentum

        # new Hamiltonian
        log_post_new = model_log_posterior(params_new)
        new_hamiltonian = -log_post_new + 0.5 * np.sum(momentum**2)

        # Metropolis acceptance criterion
        if np.log(np.random.rand()) <  hamiltonian - new_hamiltonian:
            params = params_new
            log_post = log_post_new
            accepted += 1

        params_chain.append(params)
        log_post_chain.append(log_post)

    chains = pd.DataFrame({name: chain for name, chain in zip(names, np.array(params_chain).T)})
    chains["log post"] = log_post_chain

    return chains, accepted / n_steps
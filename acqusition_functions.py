import numpy as np
from scipy.stats import norm


# Acquisition functions
def probability_of_improvement(current_best, mean, std, xi):
    '''
    Thus function implements the probability of improvement acquisition function.
    It implements
        PI(x) = P(f(x) >= f(x^+))
            since we consider f(x^+) = mu^+ + epxis we have
              = Phi ( (mu - mu^+ - xi) / sigma )
    :param current_best: this is the current max of the unknown function: mu^+
    :param mean: this is the mean function from the GP over the considered set of points
    :param std: this is the std function from the GP over the considered set of points
    :param xi: small value added to avoid corner case
    :return: the value of this acquisition function for all the points
    '''
    # since std coan be 0, to avoid an error, add a small value in the denominator (like +1e-9)
    PI = norm.cdf( (mean - current_best - xi) / (std + 1e-9) ) #the norm.cdf function is your friend 
    return PI 


def expected_improvement(current_best, mean, std, xi):
    '''
    It implements the following function:

            | (mu - mu^+ - xi) Phi(Z) + sigma phi(Z) if sigma > 0
    EI(x) = |
            | 0                                       if sigma = 0

            where Phi is the CDF and phi the PDF of the normal distribution
            and
            Z = (mu - mu^+ - xi) / sigma

    :param current_best: this is the current max of the unknown function: mu^+
    :param mean: this is the mean function from the GP over the considered set of points
    :param std: this is the std function from the GP over the considered set of points
    :param xi: small value added to avoid corner case
    :return: the value of this acquisition function for all the points
    '''

    # start by computing the Z as we did in the probability of improvement function
    # to avoid division by 0, add a small term eg. np.spacing(1e6) to the denominator
    Z = (mean - current_best - xi) / (std + 1e-9)
    # now we have to compute the output only for the terms that have their std > 0
    EI = (mean  - current_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0] = 0
    
    return EI



def GP_UCB(mean, std, t, dim = 1.0, v = 1.0, delta = .1):
    '''
    Implementation of the Gaussian Process - Upper Confident Bound:
        GP-UBC(x) = mu + sqrt(v * beta) * sigma

    where we are usinv v = 1 and beta = 2 log( t^(d/2 + 2) pi^2 / 3 delta)
    as proved in Srinivas et al, 2010, to have 0 regret.

    :param mean: this is the mean function from the GP over the considered set of points
    :param std: this is the std function from the GP over the considered set of points
    :param t: iteration number
    :param dim: dimension of the input space
    :param v: hyperparameter that weights the beta for the exploration-exploitation trade-off. If v = 1 and another
              condition, it is proved we have 0 regret
    :param delta: hyperparameter used in the computation of beta
    :return: the value of this acquisition function for all the points
    '''
    beta =  2 * np.log( t**(dim/2 + 2) * np.pi**2 / (3 * delta + 1e-9)  ) 
    UCB = mean + np.sqrt(v * beta) * std
    
    return UCB



def GP_UCB_approx(mean, std, t, eps):
    '''
    Implementation of the Gaussian Process - Upper Confident Bound in a easy approximate way:
        GP-UBC(x) = mu + eps * log(t) * sigma

    we use the fact that beta ~ log(t)^2, so we have sqrt(v * log(t)^2) = log(t)*sqrt(v) ~ eps * log(t)

    :param mean: this is the mean function from the GP over the considered set of points
    :param std: this is the std function from the GP over the considered set of points
    :param t: iteration number
    :param eps: trade-off constant
    :return: the value of this acquisition function for all the points
    '''
    UCB = mean + eps * np.log(t) * std
    return UCB 

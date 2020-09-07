import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg
import torch
from torch import nn, distributions
from tqdm import tqdm


def squared_exponential_kernel(x, y, lengthscale, variance):
    '''
    Function that computes the covariance matrix using a squared-exponential kernel
    '''
    # pair-wise distances, size: NxM
    sqdist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean')
    
    # compute the kernel
    cov_matrix = variance * np.exp(-0.5 * sqdist * (1/lengthscale**2))  # NxM
    return cov_matrix


def fit_predictive_GP(X, y, Xtest, lengthscale, kernel_variance, noise_variance):
    '''
    Function that fit the Gaussian Process. It returns the predictive mean function and
    the predictive covariance function. It follows step by step the algorithm on the lecture
    notes
    '''
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    K = squared_exponential_kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X))) #

    # compute the mean at our test points.
    Ks = squared_exponential_kernel(X, Xtest, lengthscale, kernel_variance)
    alpha = linalg.solve(L.transpose(), linalg.solve(L,y)) #
    mu = np.dot(np.transpose(Ks), alpha) #

    v = linalg.solve(L,Ks)
    # compute the variance at our test points.
    Kss = squared_exponential_kernel(Xtest, Xtest, lengthscale, kernel_variance)
    covariance = Kss - np.dot(v.transpose(), v)
    
    return mu, covariance


# Here PyTorch is used to define the optimization function, with an ADAM optimizer
# This is likely not a very efficient nor pretty way to do this.

def optimize_GP_hyperparams(Xtrain, ytrain, optimization_steps, learning_rate, mean_prior, prior_std):
    '''
    Methods that run the optimization of the hyperparams of our GP. We will use
    Gradient Descent because it takes to much time to run grid search at each step
    of bayesian optimization. We use a different definition of the kernel to make the
    optimization more stable

    :param X: training set points
    :param y: training targets
    :return: values for lengthscale, output_var, noise_var that maximize the log-likelihood
    '''
    
    # we are re-defining the kernel because we need it in PyTorch
    def squared_exponential_kernel_torch(x, y, _lambda, variance):
        x = x.squeeze(1).expand(x.size(0), y.size(0))
        y = y.squeeze(0).expand(x.size(0), y.size(0))
        sqdist = torch.pow(x - y, 2)
        k = variance * torch.exp(-0.5 * sqdist * (1/_lambda**2))  # NxM
        return k

    X = np.array(Xtrain).reshape(-1,1)
    y = np.array(ytrain).reshape(-1,1)
    N = len(X)

    # tranform our training set in Tensor
    Xtrain_tensor = torch.from_numpy(X).float()
    ytrain_tensor = torch.from_numpy(y ).float()
    # we should define our hyperparameters as torch parameters where we keep track of
    # the operations to get hte gradients from them
    _lambda = nn.Parameter(torch.tensor(1.), requires_grad=True)
    output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
    noise_variance = nn.Parameter(torch.tensor(.5), requires_grad=True)

    # we use Adam as optimizer
    optim = torch.optim.Adam([_lambda, output_variance, noise_variance], lr=learning_rate)

    # optimization loop using the log-likelihood that involves the cholesky decomposition 
    nlls = []
    lambdas = []
    output_variances = []
    noise_variances = []
    for i in range(optimization_steps):
        assert noise_variance >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()
        
        # negative log-likelihood: log p(y|X) = log(2π)N/2 + y.t*a/2 + sum(log(diag(L)))
        K = squared_exponential_kernel_torch(Xtrain_tensor, Xtrain_tensor, _lambda, output_variance) 
        K += noise_variance * torch.eye(N)
        
        L = torch.cholesky(K)
        _alpha_temp, _ = torch.solve(ytrain_tensor, L)
        _alpha, _ = torch.solve(_alpha_temp, L.t())
        nll = torch.log(torch.tensor(2 * np.pi)) * N / 2 + 0.5 * torch.matmul(ytrain_tensor.transpose(0, 1),
                                                                              _alpha) + torch.sum(torch.log(torch.diag(L)))

        # log-likelihood of the prior p(l) = -log(l) + log( N(ln(l) | mu, sigma^2) )
        norm = distributions.Normal(loc=mean_prior, scale=prior_std)
        prior_negloglike =  torch.log(_lambda) - torch.log(torch.exp(norm.log_prob(_lambda)))

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(_lambda.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # projected in the constraints (lengthscale and output variance should be positive)
        for p in [_lambda, output_variance]:
            p.data.clamp_(min=0.0000001)

        noise_variance.data.clamp_(min=1e-5, max= 0.05)

        
    return _lambda.item(), output_variance.item(), noise_variance.item()


def sample_initial_parameters(n_samples, search_space, method='uniform'):
    '''
    Sample initial data points

    :param n_samples: the number of initial data points to sample
    :param size_search_space: search space to sample data points from
    :return: initial data points in X values and y values
    '''
    
    # array for the points x' and repective f(x') that we sample using the acquisition function
    X_sample = []
    
    if method == 'uniform':
        # sample uniformly distinctly data points
        x_samples = np.linspace(0,len(search_space)-1, n_samples)
        x_samples = [int(x) for x in x_samples]

        for i in tqdm(x_samples):
            xt = search_space[i]
            X_sample.append(xt)

    elif method == 'random':
        # get some random samples
        for _ in tqdm(range(n_samples)):
            xt = search_space[np.random.randint(0, size_search_space)]
            X_sample.append(xt)
            
    return X_sample


def fit_predictive_GP_multiple(X, y, Xtest, lengthscale, kernel_variance, noise_variance):
    '''
    Function that fit the Gaussian Process. It returns the predictive mean function and
    the predictive covariance function. It follows step by step the algorithm on the lecture
    notes
    '''
    
    def squared_exponential_kernel(x, y, lengthscale, variance):
        '''
        Function that computes the covariance matrix using a squared-exponential kernel
        '''
        # pair-wise distances, size: NxM
        sqdist = cdist(x, y, 'sqeuclidean')

        # compute the kernel
        cov_matrix = variance * np.exp(-0.5 * sqdist * (1/lengthscale**2))  # NxM
        return cov_matrix
    
    X = np.array(X)#.reshape(-1, 1)
    y = np.array(y)
    K = squared_exponential_kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X))) #

    # compute the mean at our test points.)
    Ks = squared_exponential_kernel(X, Xtest, lengthscale, kernel_variance)
    alpha = linalg.solve(L.transpose(), linalg.solve(L,y)) #
    mu = np.dot(np.transpose(Ks), alpha) #

    v = linalg.solve(L,Ks)
    # compute the variance at our test points.
    Kss = squared_exponential_kernel(Xtest, Xtest, lengthscale, kernel_variance)
    covariance = Kss - np.dot(v.transpose(), v)
    
    return mu, covariance


# Here PyTorch is used to define the optimization function, with an ADAM optimizer
# This is likely not a very efficient nor pretty way to do this.

def optimize_GP_hyperparams_multiple(Xtrain, ytrain, optimization_steps, learning_rate, mean_prior, prior_std):
    '''
    Methods that run the optimization of the hyperparams of our GP. We will use
    Gradient Descent because it takes to much time to run grid search at each step
    of bayesian optimization. We use a different definition of the kernel to make the
    optimization more stable

    :param X: training set points
    :param y: training targets
    :return: values for lengthscale, output_var, noise_var that maximize the log-likelihood
    '''
    
    # pairwise euclidean distance for two matrices
    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)
    
    # we are re-defining the kernel because we need it in PyTorch
    def squared_exponential_kernel_torch(x, y, _lambda, variance):
        k = variance * torch.exp(-0.5 * pairwise_distances(x, y) * (1/_lambda**2)) #NxN
        return k

    X = np.array(Xtrain)#.reshape(-1,1)
    y = np.array(ytrain).reshape(-1,1)
    N = X.shape[0]

    # tranform our training set in Tensor
    Xtrain_tensor = torch.from_numpy(X).float()
    ytrain_tensor = torch.from_numpy(y ).float()
    # we should define our hyperparameters as torch parameters where we keep track of
    # the operations to get hte gradients from them
    _lambda = nn.Parameter(torch.tensor(1.), requires_grad=True)
    output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
    noise_variance = nn.Parameter(torch.tensor(.5), requires_grad=True)

    # we use Adam as optimizer
    optim = torch.optim.Adam([_lambda, output_variance, noise_variance], lr=learning_rate)

    # optimization loop using the log-likelihood that involves the cholesky decomposition 
    nlls = []
    lambdas = []
    output_variances = []
    noise_variances = []
    for i in range(optimization_steps):
        assert noise_variance >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()
        
        # negative log-likelihood: log p(y|X) = log(2π)N/2 + y.t*a/2 + sum(log(diag(L)))
        K = squared_exponential_kernel_torch(Xtrain_tensor, Xtrain_tensor, _lambda, output_variance)
        K += noise_variance * torch.eye(N)
        
        L = torch.cholesky(K)
        _alpha_temp, _ = torch.solve(ytrain_tensor, L)
        _alpha, _ = torch.solve(_alpha_temp, L.t())
        nll = torch.log(torch.tensor(2 * np.pi)) * N / 2 + 0.5 * torch.matmul(ytrain_tensor.transpose(0, 1),
                                                                              _alpha) + torch.sum(torch.log(torch.diag(L)))

        # log-likelihood of the prior p(l) = -log(l) + log( N(ln(l) | mu, sigma^2) )
        norm = distributions.Normal(loc=mean_prior, scale=prior_std)
        prior_negloglike =  torch.log(_lambda) - torch.log(torch.exp(norm.log_prob(_lambda)))

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(_lambda.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # projected in the constraints (lengthscale and output variance should be positive)
        for p in [_lambda, output_variance]:
            p.data.clamp_(min=0.0000001)

        noise_variance.data.clamp_(min=1e-5, max= 0.05)

        
    return _lambda.item(), output_variance.item(), noise_variance.item()
